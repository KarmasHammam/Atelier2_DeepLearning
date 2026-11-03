"""MNIST : séparation commune, transformations adaptées et boîtes de chiffres."""

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

from .config import ROOT

MEAN, STD = (0.1307,), (0.3081,)
IMAGENET_MEAN, IMAGENET_STD = (.485, .456, .406), (.229, .224, .225)


def mnist():
    # Même format IDX MNIST standard que le jeu du sujet. Le miroir GitHub
    # évite la restriction Kaggle ; torchvision conserve ses MD5 de référence.
    datasets.MNIST.mirrors = ['https://raw.githubusercontent.com/fgnt/mnist/master/']
    root = ROOT / 'data'
    return (datasets.MNIST(root, train=True, download=True),
            datasets.MNIST(root, train=False, download=True))


def select(indices, labels, size, seed):
    """Sous-échantillonnage stratifié reproductible, sans changer le split initial."""
    indices = np.asarray(indices)
    if size >= len(indices):
        return indices
    return train_test_split(indices, train_size=size, stratify=np.asarray(labels)[indices],
                            random_state=seed)[0]


def split_indices(train, test, config):
    indices = np.arange(len(train))
    train_ids, valid_ids = train_test_split(indices, test_size=6000,
                                           stratify=train.targets.numpy(), random_state=config.seed)
    parts = {
        'train': select(train_ids, train.targets, config.train_size, config.seed),
        'validation': select(valid_ids, train.targets, config.valid_size, config.seed),
        'test': select(np.arange(len(test)), test.targets, config.test_size, config.seed),
    }
    parts['probe'] = select(parts['train'], train.targets, config.probe_size, config.seed)
    assert not set(parts['train']) & set(parts['validation'])
    # Le test provient d'un autre fichier IDX ; ses indices ont un autre espace.
    hashes = {k: hashlib.sha256(np.sort(v).astype('<i8').tobytes()).hexdigest()
              for k, v in parts.items()}
    return parts, hashes


def digit_box(image):
    """Encadrer les pixels non nuls ; coordonnées xyxy, borne haute exclusive."""
    pixels = np.asarray(image)
    ys, xs = np.nonzero(pixels > 0)
    if not len(xs):
        raise ValueError('Image vide : aucune boîte de chiffre ne peut être créée.')
    return torch.tensor([[xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]], dtype=torch.float32)


class Digits(Dataset):
    def __init__(self, dataset, indices, model, augment=False):
        self.dataset, self.indices, self.model = dataset, np.asarray(indices), model
        self.augment = transforms.RandomAffine(degrees=10, translate=(.05, .05)) if augment else None
        if model in ('vgg16', 'alexnet'):
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)), transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        elif model == 'fasterrcnn':
            # Le détecteur fait lui-même normalisation et redimensionnement.
            self.transform = transforms.Compose([transforms.Grayscale(num_output_channels=3),
                                                  transforms.ToTensor()])
        else:
            self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        original = int(self.indices[index])
        image, label = self.dataset[original]
        if self.augment:
            image = self.augment(image)
        if self.model == 'fasterrcnn':
            # Recalculer la boîte APRÈS la transformation géométrique.
            box = digit_box(image)
            target = {'boxes': box, 'labels': torch.tensor([label + 1], dtype=torch.int64),
                      'image_id': torch.tensor(original),
                      'area': (box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1]),
                      'iscrowd': torch.zeros(1, dtype=torch.int64)}
            # 0 est réservé au fond : le chiffre 0 correspond à la classe 1.
            return self.transform(image), target
        return self.transform(image), label


def detection_collate(batch):
    return tuple(zip(*batch))


def loaders(config):
    train, test = mnist()
    indices, hashes = split_indices(train, test, config)
    result = {}
    for name, ids in indices.items():
        dataset = Digits(test if name == 'test' else train, ids, config.model,
                         augment=config.augmentation and name == 'train')
        result[name] = DataLoader(dataset, batch_size=config.batch_size,
                                  shuffle=name == 'train', num_workers=0,
                                  generator=torch.Generator().manual_seed(config.seed),
                                  collate_fn=detection_collate if config.model == 'fasterrcnn' else None)
    metadata = {'split_hashes': hashes, 'sizes': {k: len(v) for k, v in indices.items()},
                'source': 'MNIST standard IDX, miroir fgnt/mnist ; MD5 torchvision conservés',
                'resources_md5': dict(datasets.MNIST.resources),
                'train_class_counts': np.bincount(train.targets[indices['train']].numpy(), minlength=10).tolist()}
    return result, metadata
