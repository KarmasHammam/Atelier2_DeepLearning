"""Exporter les sorties d'une exécution Colab, sans poids ni identifiants."""

import argparse
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.compare import compare


def export(profile='full'):
    compare(profile)  # Refuser une archive présentée comme complète s'il manque un modèle.
    path = ROOT / f'resultats_atelier2_{profile}.zip'
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for folder in [ROOT / 'results' / profile, ROOT / 'results' / 'eda']:
            for file in sorted(folder.rglob('*')):
                if file.is_file():
                    archive.write(file, file.relative_to(ROOT))
    print(path)
    return path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', choices=['full', 'smoke'], default='full')
    export(parser.parse_args().profile)
