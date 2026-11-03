"""Réglages explicites : une vérification CPU n'est pas une expérience GPU."""

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('cnn', 'fasterrcnn', 'vgg16', 'alexnet', 'vit')
SEED = 42


@dataclass(frozen=True)
class Config:
    model: str
    profile: str = 'full'
    seed: int = SEED
    epochs: int = 8
    batch_size: int = 64
    lr: float = .001
    weight_decay: float = .0001
    train_size: int = 54000
    valid_size: int = 6000
    test_size: int = 10000
    probe_size: int = 1000
    augmentation: bool = True
    pretrained: bool = False

    def digest(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


def configuration(model, profile='full'):
    if model not in MODELS or profile not in ('full', 'smoke'):
        raise ValueError('Modèle ou profil non reconnu.')
    options = {
        'cnn': dict(epochs=8, batch_size=64, lr=.001),
        'fasterrcnn': dict(epochs=3, batch_size=8, lr=.005, pretrained=True),
        'vgg16': dict(epochs=3, batch_size=32, lr=.0001, pretrained=True),
        'alexnet': dict(epochs=3, batch_size=64, lr=.0001, pretrained=True),
        'vit': dict(epochs=12, batch_size=128, lr=.0003, weight_decay=.01),
    }[model]
    if profile == 'smoke':
        # Exécuter le vrai modèle préentraîné et des gradients sur un petit jeu.
        # Ces scores très courts servent à diagnostiquer le code, pas à classer les modèles.
        options |= dict(epochs=1, batch_size=2 if model == 'fasterrcnn' else 4,
                        train_size=64, valid_size=40, test_size=80, probe_size=40,
                        augmentation=False)
    return Config(model=model, profile=profile, **options)
