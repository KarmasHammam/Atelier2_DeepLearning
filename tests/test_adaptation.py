"""Vérifier l'import attribué et l'adaptation sans simuler un entraînement GPU."""
import ast
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import nbformat
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import ROOT
from src.adaptation import save_result
from scripts import export_adapted_results as exporter


class AdaptationTests(unittest.TestCase):
    def test_original_notebooks_preserve_recorded_outputs_and_hashes(self):
        folder = ROOT / 'references/eachr4f'
        manifest = json.loads((folder / 'manifest.json').read_text())
        self.assertEqual(len(manifest['original_files_sha256']), 5)
        for name, expected in manifest['original_files_sha256'].items():
            path = folder / 'notebooks' / name
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)
            notebook = nbformat.read(path, as_version=4)
            self.assertTrue(any(c.get('outputs') for c in notebook.cells if c.cell_type == 'code'))

    def test_corrected_notebooks_have_no_stale_outputs_and_compile(self):
        files = list((ROOT / 'notebooks/colab').glob('*.ipynb'))
        self.assertEqual(len(files), 5)
        workers = 0
        for path in files:
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            for cell in notebook.cells:
                if cell.cell_type != 'code':
                    continue
                self.assertIsNone(cell.execution_count)
                self.assertEqual(cell.outputs, [])
                tree = ast.parse(cell.source)
                compile(tree, str(path), 'exec')
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'worker_script' for t in node.targets):
                        if isinstance(node.value, ast.Constant):
                            worker = node.value.value
                            compile(worker, str(path) + ':worker', 'exec')
                            self.assertIn('ATELIER2_WORK_DIR', worker)
                            if 'faster' in path.name:
                                self.assertIn('value=detection_loss(', worker)
                            workers += 1
        self.assertEqual(workers, 3)

    def definitions(self, name):
        notebook = nbformat.read(ROOT / 'notebooks/colab' / name, as_version=4)
        nodes = []
        for cell in notebook.cells:
            if cell.cell_type == 'code':
                nodes.extend(n for n in ast.parse(cell.source).body if isinstance(n, (ast.ClassDef, ast.FunctionDef)))
        namespace = {'torch': torch, 'nn': nn, 'np': np}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), name, 'exec'), namespace)
        return notebook, namespace

    def test_adapted_cnn_and_vit_forward_backward(self):
        _, cnn = self.definitions('01_cnn.ipynb')
        _, vit = self.definitions('05_vit.ipynb')
        models = [cnn['MNIST_CNN'](), vit['MyViT']((1, 28, 28))]
        for model in models:
            logits = model(torch.randn(3, 1, 28, 28))
            self.assertEqual(tuple(logits.shape), (3, 10))
            nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2])).backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_vit_final_loss_weights_last_incomplete_batch(self):
        notebook, _ = self.definitions('05_vit.ipynb')
        source = next(c.source for c in notebook.cells if c.cell_type == 'code' and '# Évaluation finale' in c.source)
        # 17 exemples en lots de 8 : une moyenne simple des trois lots est fausse.
        logits = torch.zeros(17, 10)
        logits[-1, 0] = -10
        labels = torch.zeros(17, dtype=torch.long)
        loader = DataLoader(TensorDataset(logits, labels), batch_size=8)
        namespace = {'model': nn.Identity(), 'device': torch.device('cpu'),
                     'torch': torch, 'criterion': nn.CrossEntropyLoss(),
                     'test_loader': loader, 'tqdm': lambda iterable, **kwargs: iterable}
        with contextlib.redirect_stdout(io.StringIO()):
            exec(compile(source, 'vit_final_evaluation', 'exec'), namespace)
        expected = nn.functional.cross_entropy(logits, labels).item()
        self.assertAlmostEqual(namespace['test_loss'], expected, places=6)

    def test_export_distinguishes_partial_and_imported_results(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            folder = root / 'results/adapted_gpu/cnn'
            folder.mkdir(parents=True)
            path = folder / 'result.json'
            data = {'status': 'completed', 'origin': 'new_execution_of_adapted_notebook',
                    'protocol': 'common', 'device': 'test fixture'}
            path.write_text(json.dumps(data))
            with patch.object(exporter, 'ROOT', root):
                with self.assertRaisesRegex(ValueError, 'incomplète'):
                    exporter.export()
                with contextlib.redirect_stdout(io.StringIO()):
                    archive = exporter.export(allow_partial=True)
                with zipfile.ZipFile(archive) as file:
                    manifest = json.loads(file.read('export_manifest.json'))
                    self.assertFalse(manifest['complete'])
                    self.assertEqual(len(manifest['missing']), 4)
                data['origin'] = 'imported_reference'
                path.write_text(json.dumps(data))
                with self.assertRaisesRegex(ValueError, 'importés'):
                    exporter.export(allow_partial=True)
                data.update(origin='new_execution_of_adapted_notebook', accuracy=.9,
                            f1_macro=.9, test_loss=.1, training_seconds=1., gpu_count_used_for_training=1)
                for name in exporter.MODELS:
                    path = root / 'results/adapted_gpu' / name / 'result.json'
                    path.parent.mkdir(exist_ok=True)
                    path.write_text(json.dumps(data | {'model': name}))
                with contextlib.redirect_stdout(io.StringIO()):
                    archive = exporter.export()
                with zipfile.ZipFile(archive) as file:
                    manifest = json.loads(file.read('export_manifest.json'))
                    self.assertTrue(manifest['complete'])
                    self.assertIn('results/adapted_gpu/comparison.csv', file.namelist())

    def test_save_result_refuses_cpu_as_gpu_evidence(self):
        with patch('torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'GPU réelle'):
                save_result('cnn', [], [], 0., 0., ROOT)


if __name__ == '__main__':
    unittest.main()
