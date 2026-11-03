"""CNN, Faster R-CNN, transfert et ViT : architectures commentées."""

import os
import math

import torch
from torch import nn
from torchvision import models
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.anchor_utils import AnchorGenerator

from .config import ROOT

os.environ.setdefault('TORCH_HOME', str(ROOT / '.cache' / 'torch'))


class CNN(nn.Module):
    """28×28 → 14×14 → 7×7 ; stride=1 et padding=1 dans les convolutions."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(64 * 7 * 7, 128),
                                        nn.ReLU(), nn.Dropout(.25), nn.Linear(128, 10))

    def forward(self, x):
        return self.classifier(self.features(x))


class Attention(nn.Module):
    """Attention multi-têtes implémentée explicitement, sans modèle préentraîné."""
    def __init__(self, dim=64, heads=4, dropout=.1):
        super().__init__()
        if dim % heads:
            raise ValueError('La dimension doit être divisible par le nombre de têtes.')
        self.heads, self.head_dim = heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.projection = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, tokens, dim = x.shape
        qkv = self.qkv(x).reshape(batch, tokens, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        weights = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = self.dropout(weights.softmax(dim=-1))
        context = (weights @ v).transpose(1, 2).reshape(batch, tokens, dim)
        return self.projection(context)


class EncoderBlock(nn.Module):
    def __init__(self, dim=64, heads=4, mlp_dim=128, dropout=.1):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attention = Attention(dim, heads, dropout)
        self.mlp = nn.Sequential(nn.Linear(dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(mlp_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class ViT(nn.Module):
    """28×28 découpée en 16 patches 7×7, puis token CLS et positions apprises."""
    def __init__(self, image_size=28, patch_size=7, dim=64, depth=3, heads=4):
        super().__init__()
        if image_size % patch_size:
            raise ValueError('La taille des patches doit diviser la taille de l’image.')
        self.image_size = image_size
        patches = (image_size // patch_size) ** 2
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
        self.embedding = nn.Linear(patch_size ** 2, dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.position = nn.Parameter(torch.randn(1, patches + 1, dim) * .02)
        self.blocks = nn.Sequential(*[EncoderBlock(dim, heads) for _ in range(depth)])
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 10))

    def forward(self, image):
        if image.shape[1:] != (1, self.image_size, self.image_size):
            raise ValueError('Le ViT attend des images MNIST 1×28×28.')
        tokens = self.embedding(self.unfold(image).transpose(1, 2))
        tokens = torch.cat([self.cls.expand(image.size(0), -1, -1), tokens], dim=1)
        return self.head(self.blocks(tokens + self.position)[:, 0])


def transfer(name, pretrained=True):
    weights = (models.VGG16_Weights.IMAGENET1K_V1 if name == 'vgg16'
               else models.AlexNet_Weights.IMAGENET1K_V1) if pretrained else None
    model = models.vgg16(weights=weights) if name == 'vgg16' else models.alexnet(weights=weights)
    # torchvision vérifie le hash des poids officiels. Aucun remplacement
    # silencieux par des poids aléatoires si le téléchargement échoue.
    for parameter in model.parameters():
        parameter.requires_grad = False
    # Fine-tuning réel : dernier bloc convolutif + nouvelle tête, pas seulement
    # apprentissage d'un classifieur au-dessus de caractéristiques figées.
    for layer in list(model.features.children())[24 if name == 'vgg16' else 10:]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    inputs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(inputs, 10)
    return model


def detector(pretrained=True):
    weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1 if pretrained else None
    model = fasterrcnn_resnet50_fpn(
        weights=weights, weights_backbone=None, min_size=64, max_size=64,
        rpn_pre_nms_top_n_train=200, rpn_pre_nms_top_n_test=100,
        rpn_post_nms_top_n_train=100, rpn_post_nms_top_n_test=50,
        box_detections_per_img=20, box_score_thresh=0.)
    inputs = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(inputs, 11)
    # Ancres plus petites pour des chiffres ; toujours trois ratios par niveau,
    # donc compatibles avec la forme des poids du RPN préentraîné.
    model.rpn.anchor_generator = AnchorGenerator(
        sizes=((8,), (16,), (32,), (64,), (128,)), aspect_ratios=((.5, 1., 2.),) * 5)
    return model


def build_model(name, pretrained=True):
    if name == 'cnn':
        return CNN()
    if name == 'vit':
        return ViT()
    if name in ('vgg16', 'alexnet'):
        return transfer(name, pretrained)
    if name == 'fasterrcnn':
        return detector(pretrained)
    raise ValueError(name)
