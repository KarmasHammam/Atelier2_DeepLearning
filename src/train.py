"""Entraîner un modèle et écrire les résultats ; full exige un vrai GPU CUDA."""

import argparse
from dataclasses import asdict
import gc
import hashlib
import json
import os
import platform
import time

os.environ.setdefault('MPLCONFIGDIR', '/tmp/atelier2-matplotlib')
os.environ.setdefault('XDG_CACHE_HOME', '/tmp/atelier2-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from sklearn.metrics import classification_report

from .config import ROOT, MODELS, configuration
from .data import loaders
from .engine import seed_everything, train_epoch, evaluate
from .models import build_model


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def save_plots(folder, history, predictions, profile):
    history = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.epoch, history.train_loss, label='Train (tous les lots)')
    axes[0].plot(history.epoch, history.validation_loss, label='Validation')
    axes[0].set_title('Loss : diagnostic propre à l’architecture')
    axes[1].plot(history.epoch, history.train_accuracy_probe, label='Échantillon fixe du train')
    axes[1].plot(history.epoch, history.validation_accuracy, label='Validation')
    axes[1].set_title('Accuracy par image')
    for ax in axes:
        ax.set_xlabel('Époque')
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle('VÉRIFICATION CPU — pas un résultat final' if profile == 'smoke' else 'Entraînement complet GPU')
    fig.tight_layout()
    fig.savefig(folder / 'learning_curves.png', dpi=140)
    plt.close(fig)
    include_empty = any(row['predicted'] == -1 for row in predictions)
    labels = [-1] + list(range(10)) if include_empty else list(range(10))
    matrix = np.zeros((10, len(labels)), dtype=int)
    for row in predictions:
        matrix[row['true'], labels.index(row['predicted'])] += 1
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(matrix, cmap='Blues')
    for (y, x), value in np.ndenumerate(matrix):
        ax.text(x, y, str(value), ha='center', va='center', fontsize=7,
                color='white' if value > matrix.max() / 2 else 'black')
    ax.set_xticks(range(len(labels)), ['Aucune' if value == -1 else str(value) for value in labels])
    ax.set_yticks(range(10), range(10))
    ax.set(xlabel='Chiffre prédit', ylabel='Chiffre réel', title='Test : matrice de confusion par image')
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(folder / 'confusion_matrix.png', dpi=140)
    plt.close(fig)
    pd.DataFrame(matrix, index=range(10), columns=labels).to_csv(folder / 'confusion_matrix.csv')


def run(model_name, profile='full', device_name='auto', force=False):
    config = configuration(model_name, profile)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device_name == 'auto' else torch.device(device_name)
    if profile == 'full' and (device.type != 'cuda' or not torch.cuda.is_available()):
        raise RuntimeError('Le profil full exige un GPU CUDA. Dans Colab : Exécution > Modifier le type d’exécution > GPU. Utiliser --profile smoke --device cpu pour une vérification locale.')
    if profile == 'smoke' and device.type != 'cpu':
        raise ValueError('Le profil smoke de ce projet est explicitement une vérification CPU.')
    seed_everything(config.seed)
    folder = ROOT / 'results' / profile / model_name
    folder.mkdir(parents=True, exist_ok=True)
    data, metadata = loaders(config)
    digest = hashlib.sha256()
    for filename in ['config.py', 'data.py', 'models.py', 'engine.py', 'train.py']:
        digest.update((ROOT / 'src' / filename).read_bytes())
    code_digest = digest.hexdigest()
    result_path = folder / 'result.json'
    if result_path.exists() and not force:
        previous = json.loads(result_path.read_text())
        if (previous['config_digest'] == config.digest() and
                previous['data']['split_hashes'] == metadata['split_hashes'] and
                previous['code_digest'] == code_digest):
            print(f'{model_name} : résultat existant réutilisé ({profile}).', flush=True)
            return previous
        raise RuntimeError('Résultat existant incompatible. Archiver ce dossier puis utiliser --force pour le remplacer.')
    print(f'{model_name} | {profile} | {device} | tailles {metadata["sizes"]}', flush=True)
    network = build_model(model_name, pretrained=config.pretrained).to(device)
    trainable = [p for p in network.parameters() if p.requires_grad]
    if model_name == 'fasterrcnn':
        optimizer = torch.optim.SGD(trainable, lr=config.lr, momentum=.9, weight_decay=config.weight_decay)
    else:
        optimizer = torch.optim.AdamW(trainable, lr=config.lr, weight_decay=config.weight_decay)
    checkpoint = ROOT / 'checkpoints' / profile / f'{model_name}.pt'
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    # --force remplace une exécution. Ne pas laisser un ancien statut completed
    # si le nouvel entraînement s'interrompt après avoir écrit un historique partiel.
    result_path.unlink(missing_ok=True)
    history, best_score, best_epoch = [], -1., 0
    training_seconds, validation_seconds = 0., 0.
    detection = model_name == 'fasterrcnn'
    for epoch in range(1, config.epochs + 1):
        loss, seconds = train_epoch(network, data['train'], optimizer, device, detection)
        validation, _ = evaluate(network, data['validation'], device, detection)
        probe, _ = evaluate(network, data['probe'], device, detection, calculate_loss=False)
        training_seconds += seconds
        validation_seconds += validation['evaluation_seconds']
        history.append({'epoch': epoch, 'train_loss': loss,
                        'train_accuracy_probe': probe['accuracy'],
                        'validation_loss': validation['loss'],
                        'validation_accuracy': validation['accuracy'],
                        'validation_f1_macro': validation['f1_macro'],
                        'training_seconds': seconds})
        if validation['f1_macro'] > best_score:
            best_score, best_epoch = validation['f1_macro'], epoch
            # Sauvegarde CPU : pas de copie permanente d'un deuxième modèle sur GPU.
            torch.save({k: v.detach().cpu() for k, v in network.state_dict().items()}, checkpoint)
        pd.DataFrame(history).to_csv(folder / 'history.csv', index=False)
        print(f'  époque {epoch}/{config.epochs} loss={loss:.4f} val_acc={validation["accuracy"]:.4f} val_f1={validation["f1_macro"]:.4f}', flush=True)
    # Aucun test n'a été lu avant la sélection du checkpoint sur validation.
    network.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    test, predictions = evaluate(network, data['test'], device, detection)
    validation, _ = evaluate(network, data['validation'], device, detection)
    train_probe, _ = evaluate(network, data['probe'], device, detection)
    pd.DataFrame(predictions).to_csv(folder / 'predictions.csv', index=False)
    report = classification_report([row['true'] for row in predictions],
                                    [row['predicted'] for row in predictions], labels=list(range(10)),
                                    output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(folder / 'classification_report.csv')
    save_plots(folder, history, predictions, profile)
    result = {'status': 'completed', 'profile': profile, 'model': model_name,
              'purpose': 'CPU functional smoke check, NOT final GPU evidence' if profile == 'smoke' else 'Full MNIST GPU experiment',
              'config': asdict(config), 'config_digest': config.digest(), 'code_digest': code_digest,
              'device_type': device.type,
              'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor() or 'CPU',
              'cuda_runtime': torch.version.cuda, 'python': platform.python_version(),
              'torch': torch.__version__, 'torchvision': torchvision.__version__,
              'parameters': sum(p.numel() for p in network.parameters()),
              'trainable_parameters': sum(p.numel() for p in trainable),
              'selected_epoch': best_epoch, 'training_seconds': training_seconds,
              'validation_seconds': validation_seconds, 'test': test,
              'validation': validation, 'train_probe': train_probe, 'data': metadata,
              'pretrained_weights': ('COCO_V1' if detection else 'IMAGENET1K_V1') if config.pretrained else None,
              'timing_scope': 'Synchronized training loops incl. loading/transforms; excludes downloads, model creation, validation/probe/test and checkpoint writing.'}
    save_json(result_path, result)
    print(f'  TEST ({profile}) accuracy={test["accuracy"]:.4f} f1={test["f1_macro"]:.4f}', flush=True)
    del network, optimizer, trainable, data
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Atelier 2 — entraînement réel et traçable')
    parser.add_argument('--model', choices=MODELS + ('all',), default='all')
    parser.add_argument('--profile', choices=('full', 'smoke'), default='full')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    for name in MODELS if args.model == 'all' else [args.model]:
        run(name, args.profile, args.device, args.force)
