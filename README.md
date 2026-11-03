# Atelier 2 — MNIST : CNN, Faster R-CNN, transfert et ViT

**Adaptation et présentation : KARMAS Hammam**
**Base importée : [Eachr4f/atelier2-mnist](https://github.com/Eachr4f/atelier2-mnist)**,
commit `47654402f475fbe0e1d608b8e8417f28bb080d27`.

L'objectif est de reconnaître les chiffres manuscrits de **0 à 9** avec cinq modèles
PyTorch, puis d'analyser leur accuracy, leur F1, leur loss et leur temps de calcul.

Le dépôt contient désormais les **cinq notebooks source avec leurs résultats GPU
conservés**, ainsi que **cinq versions adaptées et corrigées pour Colab**.
Les résultats de la source restent attribués à Eachr4f. Ils ne sont pas présentés
comme une nouvelle exécution de KARMAS Hammam ni comme une validation du code corrigé.

## 1. Par où commencer ?

| Modèle | Version adaptée à exécuter | Ouvrir dans Colab | Résultats source déjà visibles |
| --- | --- | --- | --- |
| CNN | [01_cnn](notebooks/colab/01_cnn.ipynb) | [Lancer](https://colab.research.google.com/github/KarmasHammam/Atelier2_DeepLearning/blob/main/notebooks/colab/01_cnn.ipynb) | [Voir](references/eachr4f/notebooks/MNIST_CNN_Result.ipynb) |
| Faster R-CNN | [02_faster_rcnn](notebooks/colab/02_faster_rcnn.ipynb) | [Lancer](https://colab.research.google.com/github/KarmasHammam/Atelier2_DeepLearning/blob/main/notebooks/colab/02_faster_rcnn.ipynb) | [Voir](references/eachr4f/notebooks/Faster_RCNN_MNIST.ipynb) |
| VGG16 | [03_vgg16](notebooks/colab/03_vgg16.ipynb) | [Lancer](https://colab.research.google.com/github/KarmasHammam/Atelier2_DeepLearning/blob/main/notebooks/colab/03_vgg16.ipynb) | [Voir](references/eachr4f/notebooks/VGG16_FineTuning_MNIST.ipynb) |
| AlexNet | [04_alexnet](notebooks/colab/04_alexnet.ipynb) | [Lancer](https://colab.research.google.com/github/KarmasHammam/Atelier2_DeepLearning/blob/main/notebooks/colab/04_alexnet.ipynb) | [Voir](references/eachr4f/notebooks/AlexNet_FineTuning_MNIST.ipynb) |
| ViT | [05_vit](notebooks/colab/05_vit.ipynb) | [Lancer](https://colab.research.google.com/github/KarmasHammam/Atelier2_DeepLearning/blob/main/notebooks/colab/05_vit.ipynb) | [Voir](references/eachr4f/notebooks/ViT_MNIST.ipynb) |

Les notebooks de la colonne **« Résultats source »** sont des copies inchangées,
avec leurs graphiques et sorties enregistrées. Les versions **« adaptées »** ont
leurs sorties effacées : elles doivent être réexécutées pour mesurer l'effet des corrections.

Les deux notebooks historiques à la racine de `notebooks/` et les modifications
Colab déjà enregistrées dans ce dépôt ont été conservés.

## 2. Les résultats GPU importés

Ces valeurs sont **lues dans les sorties du dépôt source**. Aucun de ces entraînements
n'a été reproduit indépendamment pendant l'intégration.

| Modèle | Accuracy test | F1 macro | Loss test affichée | Durée annoncée | GPU annoncés |
| --- | ---: | ---: | ---: | --- | --- |
| CNN | **99,56 %** | 0,9956 | 0,0125 | 339,7 s ; 28 époques | 1 Tesla T4 |
| Faster R-CNN | 99,22 % | 0,9921 | Non rapportée | 79,9 min ; 8 époques | 2 Tesla T4 |
| VGG16 | 99,49 % | 0,9949 | 0,0226 | 6,0 min ; 9 époques | 2 Tesla T4 |
| AlexNet | 99,25 % | 0,9924 | 0,0236 | 3,8 min ; 10 époques | 2 Tesla T4 |
| ViT | 87,92 % | **0,8772** | 0,3746¹ | 62,6 s ; 5 époques | 2 Tesla T4 |

¹ La loss finale ViT est moyennée par lot dans la source, sans pondérer le dernier
lot plus petit. La loss de fin d'époque y est 0,3776. Le code adapté corrige ce calcul.
Le F1 ViT est bien présent dans les sorties, même si le README source le dit absent.

Les données structurées et leur provenance sont dans
[results/imported_eachr4f/](results/imported_eachr4f/).

### Comment lire cette comparaison ?

Le CNN obtient la meilleure accuracy **dans les sorties disponibles**. Son avance
sur VGG16 représente sept images sur 10 000 : cela ne démontre pas une supériorité
 générale. Les conditions diffèrent : 54 000/6 000 images train/validation pour CNN,
59 000/1 000 pour Faster R-CNN, VGG16 et AlexNet, et 60 000 sans validation distincte
pour le ViT. Les budgets et le nombre de GPU diffèrent également.

Le ViT est très compact : deux blocs, huit dimensions, deux têtes et cinq époques.
Son score ne permet pas de conclure que les Transformers sont généralement moins
bons que les CNN. Les temps incluent de la validation et ne constituent pas un
benchmark à matériel ou budget égaux.

## 3. Ce qui a été adapté et corrigé

### Exécution dans Colab

- Les chemins imposés `/kaggle/input` sont remplacés par un téléchargement de MNIST.
- Les IDX standard sont fournis par le miroir GitHub `fgnt/mnist`, avec les MD5
  torchvision conservés. Aucune clé Kaggle n'est demandée.
- Les scripts DDP de Faster R-CNN, VGG16 et AlexNet acceptent **un ou deux GPU**.
- Les données et checkpoints sont écrits localement dans `data/` et `.runs/`, exclus de Git.
- Les notebooks conservent les architectures de la source et indiquent l'attribution.

### Même partage pour les cinq modèles

Les versions adaptées utilisent un tirage commun `torch.randperm(60000)` avec la
graine **42** : les 6 000 premiers indices forment la validation et les 54 000 autres
le train. Les 10 000 images du test officiel restent réservées à la fin.
Il s'agit d'un partage aléatoire commun, pas d'une stratification.
Les transformations et résolutions restent propres aux architectures.

### Faster R-CNN

Les boîtes encadrent les pixels du chiffre. La classe de détection 0 est le fond ;
les chiffres 0 à 9 utilisent les classes 1 à 10. La détection la plus confiante est
reconvertie en chiffre pour calculer les métriques par image.

Deux défauts de la source sont corrigés :

1. La loss diagnostique de validation est calculée avec les **BatchNorm figées**.
   `torch.no_grad()` seul n'empêche pas leurs statistiques de changer en mode train.
2. Une absence de détection donne **-1**, compté comme une erreur, et non le chiffre zéro.

Une loss test diagnostique est aussi calculée. Elle additionne les pertes de
classification et de localisation du détecteur : **elle n'est pas directement
comparable à l'entropie croisée des autres modèles**. L'accuracy par image ne constitue
pas une mAP et ne suffit pas à prouver la qualité des boîtes.

### ViT

Le modèle conserve l'implémentation explicite des patches, de l'attention, des
positions et des blocs encodeurs. Les corrections portent sur l'évaluation :

- une validation séparée remplace le suivi du test pendant l'entraînement ;
- les meilleurs poids sont choisis sur la loss de validation et restaurés avant le test ;
- la loss test est pondérée par le nombre d'images de chaque lot ;
- le F1 macro et les prédictions sont exportés avec les autres mesures.

**Ces changements modifient le protocole. Les anciennes valeurs ne doivent pas être
réutilisées pour prétendre mesurer les versions corrigées.**

## 4. Les architectures en quelques mots

| Modèle | Architecture et apprentissage |
| --- | --- |
| CNN | Trois convolutions 32/64/128 canaux, BatchNorm, ReLU, pooling, Dropout et tête linéaire ; apprentissage depuis zéro |
| Faster R-CNN | ResNet50-FPN et RPN, tête à 11 classes, images 64×64 ; apprentissage depuis zéro dans cette base importée |
| VGG16 | Caractéristiques ImageNet, dernier bloc convolutif ajusté, pooling 1×1 et nouvelle tête 512→10 ; images RGB 96×96 |
| AlexNet | Caractéristiques ImageNet, dernière convolution ajustée, pooling 1×1 et nouvelle tête 256→10 ; images RGB 96×96 |
| ViT | 49 patches de 4×4, token CLS, positions sinusoïdales, deux blocs à 8 dimensions et deux têtes ; apprentissage depuis zéro |

Les paramètres restent définis dans chaque notebook. Le CNN utilise au maximum
30 époques avec arrêt anticipé ; Faster R-CNN en utilise au maximum 8, VGG16 et AlexNet
10, et le ViT 5. Ces budgets sont conservés de la source, pas optimisés par GridSearch.
Le transfert bénéficie d'une connaissance ImageNet préalable : cette différence
est à prendre en compte face aux modèles entraînés depuis zéro.

## 5. Exécuter et publier ses nouveaux résultats

1. Ouvrir une version adaptée avec son lien Colab dans la section 1.
2. Choisir **Exécution → Modifier le type d'exécution → GPU**.
3. Cliquer sur **Tout exécuter**. Le notebook refuse d'enregistrer une exécution CPU
   comme preuve d'un résultat GPU.
4. La dernière cellule exporte les prédictions, les rapports par chiffre, les scores,
   le temps mesuré et le matériel dans `results/adapted_gpu/<modele>/`.
5. Télécharger l'archive proposée. Elle porte la mention **partiel** tant qu'il manque
   un des cinq modèles ; cela permet de conserver chaque expérience avant de fermer Colab.
6. Enregistrer aussi le notebook dans GitHub avec **Fichier → Enregistrer une copie
   dans GitHub**, en conservant les sorties, sous son chemin `notebooks/colab/...`.
7. Ajouter les fichiers `results/adapted_gpu/` de l'archive à ce dépôt, ou transmettre
   l'archive pour leur intégration et la mise à jour du tableau final.

Les temps des nouveaux notebooks couvrent entraînement, validation et sélection des
checkpoints, hors téléchargement et test final. Les scripts DDP écrivent leur temps
dans un fichier JSON ; aucune durée n'est inventée à partir d'une valeur absente.

Quand les cinq nouveaux modèles ont terminé :

```bash
python scripts/export_adapted_results.py
```

Pour sauvegarder les modèles déjà terminés avant de poursuivre :

```bash
python scripts/export_adapted_results.py --partial
```

Les archives de résultats excluent les poids volumineux et les identifiants d'accès.

## 6. Vérifications réalisées ici

L'environnement d'intégration ne possède pas de GPU. **Les cinq versions corrigées
n'ont donc pas été entraînées ici sur GPU.** Les vérifications locales portent sur :

- les empreintes des originaux et la conservation de leurs sorties ;
- la validité des cinq notebooks adaptés et la compilation des trois scripts DDP ;
- la propagation avant et arrière des architectures CNN et ViT adaptées sur CPU ;
- le calcul pondéré de la loss ViT, y compris avec un dernier lot incomplet ;
- la conservation des statistiques BatchNorm pendant la loss diagnostique ;
- le refus de mélanger résultats importés et nouvelles mesures, et l'état explicite des archives partielles.

Les treize tests du projet passent. La compilation des scripts DDP ne remplace pas
une exécution NCCL/CUDA : ces chemins restent à valider sur GPU.

```bash
bash scripts/setup.sh
source .venv/bin/activate
python -m unittest discover -s tests -v
```

Les outils `src.train`, les résultats `results/smoke/` et les deux notebooks historiques
à la racine restent disponibles pour le protocole initial du dépôt. Ils utilisent
d'autres architectures/réglages et ne doivent pas être confondus avec cette adaptation.
L'installation locale est CPU ; dans Colab, utiliser les cellules d'installation des
notebooks pour conserver PyTorch CUDA, et ne pas lancer `scripts/setup.sh`.

## 7. Synthèse personnelle

L'étude de ces notebooks m'aide à distinguer les approches : le CNN apprend des motifs
locaux, les modèles transférés réutilisent des caractéristiques ImageNet et le ViT
relie des patches par attention. Faster R-CNN ajoute une question de localisation,
qui nécessite de construire des boîtes pour adapter MNIST.

En comparant les sorties importées, je constate que des scores élevés ne suffisent
pas à établir une comparaison équitable. Il faut aussi regarder le partage des données,
le matériel, le nombre d'époques et la définition des pertes. Le cas du ViT montre
également l'importance de garder le test pour l'évaluation finale.

Dans cette adaptation, je retiens surtout que corriger une méthode d'évaluation
oblige à refaire les mesures. Les résultats d'origine restent utiles comme référence,
mais la conclusion sur mon protocole corrigé devra s'appuyer sur mes nouvelles
exécutions GPU.

## 8. Organisation et provenance

```text
notebooks/colab/                  Cinq versions adaptées, à réexécuter sur GPU
references/eachr4f/notebooks/     Cinq originaux avec sorties GPU conservées
references/eachr4f/PROVENANCE.md  Source, commit et différences documentées
results/imported_eachr4f/        Métriques extraites des sorties d'origine
results/adapted_gpu/             Créé par les nouvelles exécutions corrigées
src/adaptation.py               Chemins portables et enregistrement des mesures
scripts/adapt_eachr4f.py         Génération reproductible des notebooks adaptés
scripts/export_adapted_results.py Export complet ou explicitement partiel
```

Le [manifeste d'import](references/eachr4f/manifest.json) contient les SHA-256 des cinq
notebooks d'origine. Les scripts d'import ne modifient pas le dépôt source.
La remise doit inclure le code, les sorties des expériences effectivement réalisées,
le rapport README et une synthèse personnelle, avec les références utilisées.
