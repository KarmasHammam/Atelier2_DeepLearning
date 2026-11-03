"""Contrôles ciblés : splits, boîtes, attention, état du modèle et profil GPU."""

import copy
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image
import torch
from torch import nn

from src.config import configuration, MODELS
from src.data import mnist, split_indices, Digits, digit_box
from src.engine import detection_loss, image_prediction, seed_everything
from src.models import CNN, ViT
from src.compare import validate_results
from src.train import run


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        seed_everything(42)
        cls.train, cls.test = mnist()

    def test_same_split_across_all_architectures(self):
        previous = None
        for name in MODELS:
            parts, hashes = split_indices(self.train, self.test, configuration(name, 'smoke'))
            self.assertFalse(set(parts['train']) & set(parts['validation']))
            self.assertEqual(len(set(self.train.targets[parts['train']].tolist())), 10)
            if previous is not None:
                self.assertEqual(hashes, previous)
            previous = hashes

    def test_detection_target_and_label_offset(self):
        data = Digits(self.train, [0], 'fasterrcnn')
        image, target = data[0]
        self.assertEqual(tuple(image.shape), (3, 28, 28))
        self.assertEqual(int(target['labels'][0]), int(self.train.targets[0]) + 1)
        box = target['boxes'][0]
        self.assertTrue(bool(torch.all(box[2:] > box[:2])))
        raw = np.array(self.train[0][0])
        ys, xs = np.nonzero(raw)
        self.assertTrue((xs >= box[0].item()).all() and (xs < box[2].item()).all())
        self.assertTrue((ys >= box[1].item()).all() and (ys < box[3].item()).all())
        with self.assertRaises(ValueError):
            digit_box(Image.fromarray(np.zeros((28, 28), dtype=np.uint8)))

    def test_cnn_and_vit_backpropagation(self):
        image = torch.randn(4, 1, 28, 28)
        labels = torch.tensor([0, 1, 2, 3])
        for model in [CNN(), ViT()]:
            logits = model(image)
            self.assertEqual(tuple(logits.shape), (4, 10))
            nn.functional.cross_entropy(logits, labels).backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            if isinstance(model, ViT):
                self.assertEqual(tuple(model.position.shape), (1, 17, 64))
                self.assertGreater(float(model.blocks[0].attention.qkv.weight.grad.abs().sum()), 0.)

    def test_detection_empty_and_highest_confidence(self):
        output = {'scores': torch.tensor([.2, .9]), 'labels': torch.tensor([1, 10]),
                  'boxes': torch.tensor([[0., 0., 2., 2.], [1., 1., 4., 4.]])}
        prediction, score, box = image_prediction(output)
        self.assertEqual(prediction, 9)
        self.assertAlmostEqual(score, .9, places=5)
        self.assertEqual(tuple(box.shape), (1, 4))
        output = {'scores': torch.empty(0), 'labels': torch.empty(0, dtype=torch.long), 'boxes': torch.empty(0, 4)}
        self.assertEqual(image_prediction(output), (-1, 0., None))

    def test_diagnostic_loss_preserves_batchnorm_and_random_state(self):
        class DiagnosticModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.bn = nn.BatchNorm1d(3)

            def forward(self, images, targets):
                return {'loss': self.bn(torch.stack(images)).square().mean() + torch.rand(())}

        model = DiagnosticModel().eval()
        before_bn = copy.deepcopy(model.state_dict())
        images = [torch.randn(3), torch.randn(3)]
        rng = torch.get_rng_state().clone()
        value = detection_loss(model, images, [{}, {}], torch.device('cpu'))
        self.assertTrue(np.isfinite(value))
        self.assertFalse(model.training)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertTrue(all(torch.equal(before_bn[key], val) for key, val in model.state_dict().items()))

    def test_full_profile_refuses_cpu_before_training(self):
        with patch('torch.cuda.is_available', return_value=False):
            with self.assertRaisesRegex(RuntimeError, 'GPU CUDA'):
                run('cnn', 'full', 'cpu')

    def test_comparison_refuses_mixed_profiles_and_partitions(self):
        base = [{'model': name, 'status': 'completed', 'profile': 'smoke',
                 'code_digest': 'same-code',
                 'device_type': 'cpu', 'device_name': 'CPU',
                 'data': {'split_hashes': {'test': 'same'}}} for name in MODELS]
        self.assertTrue(validate_results(base, 'smoke'))
        changed = copy.deepcopy(base)
        changed[1]['data']['split_hashes']['test'] = 'different'
        with self.assertRaisesRegex(ValueError, 'Partitions'):
            validate_results(changed, 'smoke')
        with self.assertRaisesRegex(ValueError, 'profils'):
            validate_results(base, 'full')
        changed = copy.deepcopy(base)
        changed[1]['code_digest'] = 'other-code'
        with self.assertRaisesRegex(ValueError, 'protocole'):
            validate_results(changed, 'smoke')


if __name__ == '__main__':
    unittest.main()
