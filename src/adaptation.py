"""Aide aux notebooks adaptés de Eachr4f : chemins portables et exports traçables.

Les modèles restent définis dans les notebooks. Les résultats importés de la
source ne passent jamais par save_result : seuls les nouveaux calculs y entrent.
"""

from pathlib import Path
import hashlib
import json
import platform

import numpy as np
import pandas as pd
import torch
import torchvision
from sklearn.metrics import accuracy_score, f1_score, classification_report

from .config import ROOT
from .data import mnist

SOURCE = 'https://github.com/Eachr4f/atelier2-mnist'
SOURCE_COMMIT = '47654402f475fbe0e1d608b8e8417f28bb080d27'


def prepare_work(model):
    """Partager les IDX vérifiés, conserver les checkpoints hors de Git."""
    mnist()
    work = ROOT / '.runs' / model
    work.mkdir(parents=True, exist_ok=True)
    link = work / 'MNIST'
    source = ROOT / 'data' / 'MNIST'
    if not link.exists():
        link.symlink_to(source, target_is_directory=True)
    elif link.resolve() != source.resolve():
        raise RuntimeError(f'{link} contient déjà des données différentes ; ne pas les écraser.')
    return work


def save_result(model, truth, predictions, loss, seconds, work):
    """Enregistrer les métriques calculées dans la session courante, pas la source."""
    if not torch.cuda.is_available():
        raise RuntimeError('Les notebooks adaptés exigent une exécution GPU réelle.')
    folder = ROOT / 'results' / 'adapted_gpu' / model
    folder.mkdir(parents=True, exist_ok=True)
    y, p = np.asarray(truth), np.asarray(predictions)
    if len(y) != 10000 or len(p) != 10000:
        raise ValueError('Le résultat final doit porter sur les 10 000 images test officielles.')
    work = Path(work)
    timing = work / 'timing.json'
    if seconds is None and timing.exists():
        seconds = json.loads(timing.read_text())['training_seconds']
    if seconds is None:
        raise ValueError('Le temps de calcul doit être mesuré, pas estimé depuis les logs.')
    nb_file = ROOT / 'notebooks' / 'colab' / {
        'cnn': '01_cnn.ipynb', 'fasterrcnn': '02_faster_rcnn.ipynb',
        'vgg16': '03_vgg16.ipynb', 'alexnet': '04_alexnet.ipynb', 'vit': '05_vit.ipynb'}[model]
    # Empreinte des sources de cellules uniquement : les sorties peuvent ensuite
    # être enregistrées dans GitHub sans changer l'identité du programme.
    notebook = json.loads(nb_file.read_text())
    code = '\n'.join(''.join(c['source']) for c in notebook['cells'] if c['cell_type'] == 'code')
    result = {'status': 'completed', 'origin': 'new_execution_of_adapted_notebook',
              'model': model, 'author_adaptation': 'KARMAS Hammam',
              'source_repository': SOURCE, 'source_commit': SOURCE_COMMIT,
              'device': torch.cuda.get_device_name(0), 'device_count': torch.cuda.device_count(),
              'gpu_count_used_for_training': 1 if model == 'cnn' else min(2, torch.cuda.device_count()),
              'python': platform.python_version(), 'torch': torch.__version__,
              'torchvision': torchvision.__version__, 'notebook_code_sha256': hashlib.sha256(code.encode()).hexdigest(),
              'protocol': 'torch.randperm(60000, seed=42): first 6000 validation, rest 54000 train; official test 10000',
              'accuracy': float(accuracy_score(y, p)),
              'f1_macro': float(f1_score(y, p, labels=list(range(10)), average='macro', zero_division=0)),
              'test_loss': float(loss), 'training_seconds': float(seconds),
              'timing_scope': 'training + validation + checkpoint selection; excludes downloads and final test',
              'no_detection_rate': float(np.mean(p == -1))}
    pd.DataFrame({'true': y, 'predicted': p}).to_csv(folder / 'predictions.csv', index=False)
    report = classification_report(y, p, labels=list(range(10)), output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(folder / 'classification_report.csv')
    (folder / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(f'Nouvelle exécution GPU enregistrée : {folder}')
    return result
