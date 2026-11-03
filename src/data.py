"""Téléchargement de MNIST avec les MD5 de référence de torchvision."""
from torchvision import datasets
from .config import ROOT


def mnist():
    # Même format IDX MNIST standard que le jeu du sujet. Le miroir GitHub
    # évite la restriction Kaggle ; torchvision conserve ses MD5 de référence.
    datasets.MNIST.mirrors = ['https://raw.githubusercontent.com/fgnt/mnist/master/']
    root = ROOT / 'data'
    return (datasets.MNIST(root, train=True, download=True),
            datasets.MNIST(root, train=False, download=True))
