"""Exporter les nouveaux résultats adaptés, avec état partiel explicite."""

import argparse
import csv
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('cnn', 'fasterrcnn', 'vgg16', 'alexnet', 'vit')


def export(allow_partial=False):
    folder = ROOT / 'results' / 'adapted_gpu'
    available = {name: folder / name / 'result.json' for name in MODELS
                 if (folder / name / 'result.json').exists()}
    if not available:
        raise ValueError('Aucune nouvelle exécution adaptée terminée à exporter.')
    missing = [name for name in MODELS if name not in available]
    if missing and not allow_partial:
        raise ValueError('Expérience incomplète : ' + ', '.join(missing))
    results = [json.loads(path.read_text()) for path in available.values()]
    for result in results:
        if result['origin'] != 'new_execution_of_adapted_notebook' or result['status'] != 'completed':
            raise ValueError('Les résultats importés ne peuvent pas devenir de nouvelles exécutions.')
    if len({r['protocol'] for r in results}) != 1:
        raise ValueError('Les résultats ne suivent pas le même protocole.')
    manifest = {'complete': not missing, 'models': list(available), 'missing': missing,
                'origin': 'new_execution_of_adapted_notebooks',
                'same_gpu_model': len({r['device'] for r in results}) == 1,
                'timing_scope': 'training + validation + checkpoint selection',
                'note': 'Ne pas mélanger avec les résultats de référence Eachr4f ou le profil smoke.'}
    summary_files = []
    if not missing:
        rows = [{key: result.get(key) for key in ('model', 'accuracy', 'f1_macro', 'test_loss',
                                                 'training_seconds', 'device', 'gpu_count_used_for_training')}
                for result in results]
        comparison = folder / 'comparison.csv'
        with comparison.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
        report = folder / 'README.md'
        lines = ['# Nouvelles exécutions GPU des versions adaptées', '',
                 'Mesures distinctes des scores importés de Eachr4f. Les durées incluent la validation.',
                 'Les budgets et nombres de GPU peuvent différer. La loss de détection ne se classe pas avec une cross-entropy.', '',
                 '| Modèle | Accuracy | F1 macro | Loss test | Temps (s) |',
                 '| --- | --- | --- | --- | --- |']
        for row in rows:
            lines.append(f"| {row['model']} | {row['accuracy']:.4f} | {row['f1_macro']:.4f} | {row['test_loss']:.4f} | {row['training_seconds']:.2f} |")
        report.write_text('\n'.join(lines) + '\n', encoding='utf-8')
        summary_files = [comparison, report]
    filename = 'resultats_adaptes_atelier2' + ('_partiel' if missing else '_complet') + '.zip'
    destination = ROOT / filename
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED) as archive:
        archive.writestr('export_manifest.json', json.dumps(manifest, indent=2, ensure_ascii=False))
        # Un dossier partiellement entraîné peut exister sans result.json : ne
        # l'inclure que quand la mesure finale de ce modèle est terminée.
        for name in available:
            for path in sorted((folder / name).rglob('*')):
                if path.is_file():
                    archive.write(path, path.relative_to(ROOT))
        for path in summary_files:
            archive.write(path, path.relative_to(ROOT))
    print(destination)
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--partial', action='store_true')
    export(parser.parse_args().partial)
