"""Boucles PyTorch, adaptation détection/classification et mesures sans gradient."""

import random
import time

import numpy as np
import torch
from torch import nn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             precision_score, recall_score)
from torchvision.ops import box_iou


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # warn_only préserve l'exécution des opérateurs de détection CUDA dont la
    # rétropropagation peut ne pas être strictement déterministe selon la version.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def move_targets(targets, device):
    return [{key: value.to(device) for key, value in target.items()} for target in targets]


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


def train_epoch(model, loader, optimizer, device, detection=False):
    model.train()
    total, count = 0., 0
    sync(device)
    start = time.perf_counter()
    for images, labels in loader:
        if detection:
            images = [image.to(device) for image in images]
            target = move_targets(labels, device)
        else:
            images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        if detection:
            loss = sum(model(images, target).values())
        else:
            loss = nn.functional.cross_entropy(model(images), labels)
        if not torch.isfinite(loss):
            raise RuntimeError('Loss non finie : entraînement interrompu.')
        loss.backward()
        optimizer.step()
        n = len(images)
        total += float(loss.detach()) * n
        count += n
    sync(device)
    return total / count, time.perf_counter() - start


def image_prediction(output):
    """Plus haute confiance, classe 1..10 → chiffre 0..9 ; absence → -1."""
    if output['scores'].numel() == 0:
        return -1, 0., None
    index = int(output['scores'].argmax())
    return (int(output['labels'][index]) - 1, float(output['scores'][index]),
            output['boxes'][index:index + 1])


@torch.no_grad()
def evaluate(model, loader, device, detection=False, calculate_loss=True):
    model.eval()
    truth, predictions, confidence, ious = [], [], [], []
    total, count = 0., 0
    sync(device)
    start = time.perf_counter()
    for images, labels in loader:
        if detection:
            images = [image.to(device) for image in images]
            targets = move_targets(labels, device)
            # Aucune boîte ni étiquette vraie n'est fournie à la prédiction.
            outputs = model(images)
            for output, target in zip(outputs, targets):
                prediction, score, box = image_prediction(output)
                predictions.append(prediction)
                confidence.append(score)
                truth.append(int(target['labels'][0]) - 1)
                ious.append(float(box_iou(box, target['boxes'])[0, 0]) if box is not None else 0.)
            if calculate_loss:
                total += detection_loss(model, images, targets, device) * len(images)
        else:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            scores, predicted = logits.softmax(dim=1).max(dim=1)
            truth.extend(labels.cpu().tolist())
            predictions.extend(predicted.cpu().tolist())
            confidence.extend(scores.cpu().tolist())
            if calculate_loss:
                total += nn.functional.cross_entropy(logits, labels, reduction='sum').item()
        count += len(images)
    sync(device)
    elapsed = time.perf_counter() - start
    y, p = np.asarray(truth), np.asarray(predictions)
    metrics = {
        'loss': total / count if calculate_loss else None,
        'accuracy': float(accuracy_score(y, p)),
        'f1_macro': float(f1_score(y, p, labels=list(range(10)), average='macro', zero_division=0)),
        'f1_weighted': float(f1_score(y, p, labels=list(range(10)), average='weighted', zero_division=0)),
        'precision_macro': float(precision_score(y, p, labels=list(range(10)), average='macro', zero_division=0)),
        'recall_macro': float(recall_score(y, p, labels=list(range(10)), average='macro', zero_division=0)),
        'balanced_accuracy': float(balanced_accuracy_score(y, p)),
        'no_detection_rate': float(np.mean(p == -1)),
        'mean_iou': float(np.mean(ious)) if detection else None,
        'label_and_iou50_accuracy': float(np.mean((p == y) & (np.asarray(ious) >= .5))) if detection else None,
        'evaluated_images': count, 'evaluation_seconds': elapsed,
    }
    rows = [{'true': t, 'predicted': pred, 'confidence': score,
             'iou': ious[i] if detection else None}
            for i, (t, pred, score) in enumerate(zip(truth, predictions, confidence))]
    return metrics, rows
