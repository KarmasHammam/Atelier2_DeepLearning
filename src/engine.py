"""Évaluation du détecteur : loss sans mise à jour et classe par image."""

import torch
from torch import nn


def detection_loss(model, images, targets, device):
    """Loss diagnostique, pas prédiction : le détecteur exige le mode train.

    Les BatchNorm ne doivent PAS mettre à jour leurs statistiques sur validation
    ou test. Un contexte RNG isolé stabilise l'échantillonnage des propositions
    sans changer la suite des nombres aléatoires utilisés pour l'apprentissage.
    """
    was_training = model.training
    model.train()
    for layer in model.modules():
        if isinstance(layer, nn.modules.batchnorm._BatchNorm):
            layer.eval()
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(42)
            with torch.no_grad():
                return sum(model(images, targets).values()).item()
    finally:
        model.train(was_training)


def image_prediction(output):
    """Plus haute confiance, classe 1..10 → chiffre 0..9 ; absence → -1."""
    if output['scores'].numel() == 0:
        return -1, 0., None
    index = int(output['scores'].argmax())
    return (int(output['labels'][index]) - 1, float(output['scores'][index]),
            output['boxes'][index:index + 1])
