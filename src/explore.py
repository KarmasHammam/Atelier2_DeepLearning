"""Exploration visuelle MNIST sans apprendre de paramètres sur le test."""

import os
os.environ.setdefault('MPLCONFIGDIR', '/tmp/atelier2-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd
import numpy as np

from .config import ROOT, configuration
from .data import mnist, split_indices, digit_box
from .train import save_json


def explore():
    folder = ROOT / 'results' / 'eda'
    folder.mkdir(parents=True, exist_ok=True)
    train, test = mnist()
    indices, hashes = split_indices(train, test, configuration('cnn'))
    counts = pd.DataFrame({name: np.bincount((test if name == 'test' else train).targets[ids].numpy(), minlength=10)
                           for name, ids in indices.items() if name != 'probe'})
    counts.index.name = 'chiffre'
    counts.to_csv(folder / 'class_counts.csv')
    counts.plot.bar(figsize=(10, 4), title='MNIST : effectifs par chiffre et partition')
    plt.tight_layout()
    plt.savefig(folder / 'class_counts.png', dpi=140)
    plt.close()
    fig, axes = plt.subplots(3, 6, figsize=(10, 5))
    for ax, index in zip(axes.flat, indices['train'][:18]):
        image, label = train[int(index)]
        x1, y1, x2, y2 = digit_box(image)[0].tolist()
        ax.imshow(image, cmap='gray', vmin=0, vmax=255)
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='#00d6d6'))
        ax.set_title(f'Chiffre {label}', fontsize=9)
        ax.axis('off')
    fig.suptitle('Exemples du train et boîtes calculées pour Faster R-CNN')
    fig.tight_layout()
    fig.savefig(folder / 'examples_boxes.png', dpi=140)
    plt.close(fig)
    save_json(folder / 'summary.json', {
        'train_file_images': len(train), 'test_file_images': len(test), 'image_shape': [1, 28, 28],
        'raw_pixel_range': [int(train.data.min()), int(train.data.max())],
        'split_sizes': {k: len(v) for k, v in indices.items()}, 'split_hashes': hashes,
        'empty_images_train': int((train.data.flatten(1).sum(1) == 0).sum()),
        'note': 'Pas de normalisation apprise sur le test. Constantes MNIST et ImageNet fixées dans src/data.py.'})
    print('EDA enregistrée dans results/eda/.')


if __name__ == '__main__':
    explore()
