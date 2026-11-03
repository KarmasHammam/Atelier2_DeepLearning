"""Extraire les scores des sorties conservées, sans exécuter les notebooks source."""
from pathlib import Path
import csv
import json
import re

ROOT = Path(__file__).resolve().parents[1]
FILES = {'cnn': 'MNIST_CNN_Result.ipynb', 'fasterrcnn': 'Faster_RCNN_MNIST.ipynb',
         'vgg16': 'VGG16_FineTuning_MNIST.ipynb', 'alexnet': 'AlexNet_FineTuning_MNIST.ipynb',
         'vit': 'ViT_MNIST.ipynb'}
COMMIT = '47654402f475fbe0e1d608b8e8417f28bb080d27'


def extract(text, pattern):
    matches = re.findall(pattern, text, flags=re.I)
    return float(matches[-1]) if matches else None


def main():
    folder = ROOT / 'results/imported_eachr4f'
    folder.mkdir(exist_ok=True)
    rows = []
    for model, filename in FILES.items():
        notebook = json.loads((ROOT / 'references/eachr4f/notebooks' / filename).read_text())
        text = '\n'.join(''.join(o.get('text', [])) for c in notebook['cells'] if c['cell_type'] == 'code'
                         for o in c.get('outputs', []))
        accuracy = extract(text, r'Test accuracy\s*:\s*([\d.]+)')
        if model == 'vit':
            accuracy /= 100
        f1 = extract(text, r'(?:Test )?F1 macro\s*:\s*([\d.]+)')
        loss = extract(text, r'Test loss\s*:\s*([\d.]+)')
        assert accuracy is not None and f1 is not None
        rows.append({'model': model, 'accuracy': accuracy, 'f1_macro': f1, 'test_loss_reported': loss,
                     'training_time_reported': {'cnn': '339.7 seconds', 'fasterrcnn': '79.9 minutes',
                                                'vgg16': '6.0 minutes', 'alexnet': '3.8 minutes', 'vit': '62.6 seconds'}[model],
                     'epochs_reported': {'cnn': 28, 'fasterrcnn': 8, 'vgg16': 9, 'alexnet': 10, 'vit': 5}[model],
                     'gpu_count_reported': 1 if model == 'cnn' else 2,
                     'gpu_type_reported': 'Tesla T4', 'source_notebook': filename,
                     'origin': 'imported_recorded_outputs_NOT_new_execution', 'source_commit': COMMIT})
    with (folder / 'comparison.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    (folder / 'metrics.json').write_text(json.dumps({
        'source_repository': 'https://github.com/Eachr4f/atelier2-mnist', 'source_commit': COMMIT,
        'independent_gpu_reproduction_performed': False,
        'note': 'Scores lus dans les sorties conservées ; ils ne valident pas les notebooks corrigés.',
        'models': rows}, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
