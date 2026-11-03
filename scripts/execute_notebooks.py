"""Valider les notebooks en CPU : sorties smoke réelles, jamais sorties GPU simulées."""
import os
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('IPYTHONDIR', '/tmp/atelier2-ipython')
os.environ.setdefault('JUPYTER_RUNTIME_DIR', '/tmp/atelier2-jupyter')
os.environ.setdefault('MPLCONFIGDIR', '/tmp/atelier2-matplotlib')

if __name__ == '__main__':
    for path in sorted((ROOT / 'notebooks').glob('*.ipynb')):
        notebook = nbformat.read(path, as_version=4)
        client = NotebookClient(notebook, timeout=1800, kernel_name='python3',
                                resources={'metadata': {'path': str(ROOT)}})
        client.execute()
        nbformat.validate(notebook)
        nbformat.write(notebook, path)
        print(path.name, ':', sum(c.cell_type == 'code' for c in notebook.cells), 'cellules exécutées sans erreur', flush=True)
