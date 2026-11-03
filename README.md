# Atelier 2 — Deep Learning sur MNIST

**KARMAS Hammam**

Classification des chiffres 0 à 9 avec PyTorch.

## Notebooks

| Modèle | Code corrigé | Sorties de référence |
| --- | --- | --- |
| CNN | [Ouvrir](notebooks/colab/01_cnn.ipynb) | [Voir](references/eachr4f/notebooks/MNIST_CNN_Result.ipynb) |
| Faster R-CNN | [Ouvrir](notebooks/colab/02_faster_rcnn.ipynb) | [Voir](references/eachr4f/notebooks/Faster_RCNN_MNIST.ipynb) |
| VGG16 | [Ouvrir](notebooks/colab/03_vgg16.ipynb) | [Voir](references/eachr4f/notebooks/VGG16_FineTuning_MNIST.ipynb) |
| AlexNet | [Ouvrir](notebooks/colab/04_alexnet.ipynb) | [Voir](references/eachr4f/notebooks/AlexNet_FineTuning_MNIST.ipynb) |
| ViT | [Ouvrir](notebooks/colab/05_vit.ipynb) | [Voir](references/eachr4f/notebooks/ViT_MNIST.ipynb) |

## Résultats de référence importés

| Modèle | Accuracy | F1 macro | Loss test | Durée |
| --- | ---: | ---: | ---: | --- |
| CNN | 99,56 % | 0,9956 | 0,0125 | 339,7 s |
| Faster R-CNN | 99,22 % | 0,9921 | Non rapportée | 79,9 min |
| VGG16 | 99,49 % | 0,9949 | 0,0226 | 6 min |
| AlexNet | 99,25 % | 0,9924 | 0,0236 | 3,8 min |
| ViT | 87,92 % | 0,8772 | 0,3746¹ | 62,6 s |

[Provenance des résultats et du code](references/eachr4f/PROVENANCE.md).
Ces scores ne sont pas de nouvelles exécutions des versions corrigées.
¹ La moyenne de la loss ViT est corrigée dans le nouveau code.
Les partitions, budgets et nombres de GPU des références diffèrent ; les temps ne sont pas directement comparables.

## Exécution

1. Ouvrir un notebook corrigé dans Google Colab et activer un GPU.
2. Exécuter les cellules dans l'ordre. Le partage commun est 54 000 / 6 000 / 10 000, graine 42.
3. Télécharger l'archive proposée et enregistrer le notebook dans GitHub avec ses sorties.

Les nouvelles métriques sont exportées dans `results/adapted_gpu/`.
Faster R-CNN corrige la validation BatchNorm et l'absence de détection ; le ViT utilise une validation indépendante et une loss pondérée.

## Vérification locale

```bash
bash scripts/setup.sh
source .venv/bin/activate
python -m unittest discover -s tests -v
```

Les tests locaux ne remplacent pas les entraînements GPU des versions corrigées.
