"""Comparer seulement des exécutions de même profil et de mêmes partitions."""

import argparse
import json
import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/atelier2-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from .config import ROOT, MODELS


def validate_results(results, profile):
    if len(results) != len(MODELS) or {r['model'] for r in results} != set(MODELS):
        raise ValueError('Il faut les cinq modèles terminés pour établir le tableau complet.')
    reference = results[0]['data']['split_hashes']
    reference_code = results[0]['code_digest']
    for result in results:
        if result['status'] != 'completed' or result['profile'] != profile:
            raise ValueError('Exécution incomplète ou mélange de profils.')
        if result['data']['split_hashes'] != reference:
            raise ValueError('Partitions différentes : comparaison refusée.')
        if result['code_digest'] != reference_code:
            raise ValueError('Versions du protocole différentes : comparaison refusée.')
        if profile == 'full' and result['device_type'] != 'cuda':
            raise ValueError('Une exécution CPU ne remplit pas le protocole GPU.')
    # Un entraînement interrompu puis repris sur une autre carte reste valable,
    # mais ses temps ne constituent pas une comparaison sur le même matériel.
    return len({r['device_name'] for r in results}) == 1


def compare(profile='full'):
    folder = ROOT / 'results' / profile
    missing = [name for name in MODELS if not (folder / name / 'result.json').exists()]
    if missing:
        raise FileNotFoundError('Résultats manquants : ' + ', '.join(missing) + '. Exécuter les modèles sur le profil ' + profile + '.')
    results = [json.loads((folder / name / 'result.json').read_text()) for name in MODELS]
    same_hardware = validate_results(results, profile)
    rows = []
    for result in results:
        rows.append({'model': result['model'], 'profile': profile, 'device': result['device_name'],
                     'accuracy': result['test']['accuracy'], 'f1_macro': result['test']['f1_macro'],
                     'loss': result['test']['loss'],
                     'loss_type': 'detection_total' if result['model'] == 'fasterrcnn' else 'cross_entropy',
                     'training_seconds': result['training_seconds'],
                     'epochs': result['config']['epochs'], 'selected_epoch': result['selected_epoch'],
                     'parameters': result['parameters'], 'trainable_parameters': result['trainable_parameters'],
                     'test_images': result['test']['evaluated_images'],
                     'same_hardware': same_hardware})
    frame = pd.DataFrame(rows)
    frame.to_csv(folder / 'comparison.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    frame.set_index('model')[['accuracy', 'f1_macro']].plot.bar(ax=axes[0], ylim=(0, 1))
    axes[0].set_title('Scores sur les mêmes images de test')
    axes[0].set_xlabel('')
    frame.set_index('model').training_seconds.plot.bar(ax=axes[1], color='#16858d')
    axes[1].set(ylabel='Secondes', title='Boucles d’entraînement (budgets différents)', xlabel='')
    title = ('SMOKE CPU — 64 images de train, 1 époque : ne pas conclure sur les architectures'
             if profile == 'smoke' else 'MNIST complet — entraînements GPU')
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(folder / 'comparison.png', dpi=140)
    plt.close(fig)
    lines = ['# ' + title, '',
             'Les losses Faster R-CNN et les entropies croisées ont des définitions différentes : ne pas les classer ensemble.',
             'Les temps excluent téléchargement, validation, test et écriture des poids. Les budgets et tailles d’entrée diffèrent.',
             f'Même matériel déclaré pour les cinq exécutions : {same_hardware}.', '',
             '| Modèle | Accuracy | F1 macro | Loss propre au modèle | Train (s) | Époques |',
             '| --- | --- | --- | --- | --- | --- |']
    for row in rows:
        lines.append(f"| {row['model']} | {row['accuracy']:.4f} | {row['f1_macro']:.4f} | {row['loss']:.4f} | {row['training_seconds']:.2f} | {row['epochs']} |")
    (folder / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(frame.to_string(index=False))
    return frame


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', choices=('full', 'smoke'), default='full')
    compare(parser.parse_args().profile)
