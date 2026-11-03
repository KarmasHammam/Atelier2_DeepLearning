# Atelier 2 — Classification de MNIST avec PyTorch

**KARMAS Hammam**

## Objectif

Construire et comparer cinq architectures de vision pour reconnaître les chiffres
manuscrits de **0 à 9** : CNN, Faster R-CNN, VGG16, AlexNet et Vision Transformer.

## Travail réalisé

- **CNN** : trois blocs convolutionnels 32/64/128 canaux, BatchNorm, ReLU, pooling,
  Dropout et couches entièrement connectées.
- **Faster R-CNN** : ResNet50-FPN, boîtes calculées autour des chiffres et onze classes
  de détection, dont le fond. La détection la plus confiante fournit le chiffre prédit.
- **VGG16 et AlexNet** : caractéristiques ImageNet préentraînées, dernier bloc
  convolutif ajusté et nouvelle tête à dix classes ; images RGB de 96×96 pixels.
- **ViT depuis zéro** : 49 patches de 4×4 pixels, token CLS, positions sinusoïdales,
  deux blocs encodeurs, dimension 8 et attention à deux têtes.

Les versions corrigées partagent **54 000 images de train, 6 000 de validation et
10 000 de test**, avec la graine 42. Les meilleurs poids sont sélectionnés sur validation.
La validation Faster R-CNN fige les BatchNorm et compte une absence de détection
comme une erreur. Le ViT garde le test pour la fin et pondère sa loss par le nombre d'images.

## Notebooks

| Modèle | Notebook | Google Colab |
| --- | --- | --- |
| CNN | [Code](notebooks/colab/01_cnn.ipynb) | [Exécuter](https://colab.research.google.com/github/KarmasHammam/Atelier2_DeepLearning/blob/main/notebooks/colab/01_cnn.ipynb) |
| Faster R-CNN | [Code](notebooks/colab/02_faster_rcnn.ipynb) | [Exécuter](https://colab.research.google.com/github/KarmasHammam/Atelier2_DeepLearning/blob/main/notebooks/colab/02_faster_rcnn.ipynb) |
| VGG16 | [Code](notebooks/colab/03_vgg16.ipynb) | [Exécuter](https://colab.research.google.com/github/KarmasHammam/Atelier2_DeepLearning/blob/main/notebooks/colab/03_vgg16.ipynb) |
| AlexNet | [Code](notebooks/colab/04_alexnet.ipynb) | [Exécuter](https://colab.research.google.com/github/KarmasHammam/Atelier2_DeepLearning/blob/main/notebooks/colab/04_alexnet.ipynb) |
| ViT | [Code](notebooks/colab/05_vit.ipynb) | [Exécuter](https://colab.research.google.com/github/KarmasHammam/Atelier2_DeepLearning/blob/main/notebooks/colab/05_vit.ipynb) |

## Résultats disponibles et comparaison

Le tableau reprend les **exécutions importées** citées en provenance. Les versions
corrigées doivent être réexécutées sur GPU pour obtenir leurs propres mesures.

| Modèle | Accuracy test | F1 macro | Loss test affichée | Temps annoncé |
| --- | ---: | ---: | ---: | --- |
| CNN | **99,56 %** | 0,9956 | 0,0125 | 339,7 s |
| Faster R-CNN | 99,22 % | 0,9921 | Non rapportée | 79,9 min |
| VGG16 | 99,49 % | 0,9949 | 0,0226 | 6 min |
| AlexNet | 99,25 % | 0,9924 | 0,0236 | 3,8 min |
| ViT | 87,92 % | 0,8772 | 0,3746¹ | 62,6 s |

Le CNN obtient le meilleur score affiché, suivi de VGG16. Son avance représente
sept images sur 10 000 : elle ne prouve pas une supériorité générale. Le ViT compact,
entraîné seulement cinq époques, conserve une marge d'amélioration.

Les expériences importées utilisent des partitions différentes, de 5 à 28 époques
et une ou deux Tesla T4. Les durées ne sont donc pas directement comparables.
La loss Faster R-CNN combine plusieurs termes et ne se compare pas directement
à une entropie croisée. ¹ La moyenne finale de loss ViT est corrigée dans le nouveau code.

## Exécution et vérification

Dans Colab, activer un **GPU** puis exécuter les cellules dans l'ordre. MNIST est
téléchargé automatiquement avec vérification des MD5. Les scripts acceptent un ou deux GPU.
La dernière cellule exporte les nouvelles prédictions et métriques dans
`results/adapted_gpu/`. Enregistrer aussi le notebook dans GitHub avec ses sorties.

```bash
# Vérifications locales sur CPU
bash scripts/setup.sh
source .venv/bin/activate
python -m unittest discover -s tests -v
```

**Cinq tests locaux réussissent** : code des notebooks et scripts DDP, gradients CNN/ViT,
calcul de loss ViT, export et traçabilité. Les entraînements corrigés et les chemins
CUDA/DDP restent à valider sur GPU ; aucun nouveau score GPU n'est revendiqué ici.

## Synthèse personnelle

Cet atelier m'aide à comprendre les différences entre convolution, transfert
 d'apprentissage, détection et attention. Je retiens que la préparation des données,
le choix des métriques et la séparation train/validation/test sont essentiels.
Un score élevé doit être interprété avec le coût de calcul et les conditions d'entraînement.
Corriger le protocole demande de refaire les mesures avant de conclure sur les modèles.

