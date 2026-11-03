"""Créer les supports pédagogiques ; les sorties sont produites par exécution."""
from pathlib import Path
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def md(text):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text):
    return nbf.v4.new_code_cell(text.strip())


SETUP = r'''
from pathlib import Path
import importlib.util
import os
import subprocess
import sys

# Local : reconnaître la racine du dépôt ou son dossier notebooks/.
ROOT = Path.cwd()
if not (ROOT / 'src' / 'models.py').exists() and (ROOT.parent / 'src' / 'models.py').exists():
    ROOT = ROOT.parent
if not (ROOT / 'src' / 'models.py').exists():
    # Colab : le notebook ouvert depuis GitHub ne clone pas les autres fichiers.
    ROOT = Path('/content/Atelier2_DeepLearning')
    if not ROOT.exists():
        subprocess.run(['git', 'clone', 'https://github.com/KarmasHammam/Atelier2_DeepLearning.git', str(ROOT)], check=True)
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

# Conserver le PyTorch CUDA fourni par Colab ; ne jamais installer une roue CPU ici.
if importlib.util.find_spec('google') and importlib.util.find_spec('google.colab'):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements-colab.txt'])

import json
import pandas as pd
import torch
import torchvision
from IPython.display import display, Image
from src.config import configuration
from src.train import run
from src.explore import explore
from src.compare import compare

PROFILE = 'full' if torch.cuda.is_available() else 'smoke'
DEVICE = 'cuda' if PROFILE == 'full' else 'cpu'
print('PyTorch :', torch.__version__, '| torchvision :', torchvision.__version__)
print('Profil :', PROFILE, '| Matériel :', torch.cuda.get_device_name(0) if PROFILE == 'full' else 'CPU')
if PROFILE == 'smoke':
    print('ATTENTION : vérification CPU sur 64 images de train, 40 de validation, 80 de test, une époque.')
    print('Les scores affichés NE SONT PAS les résultats finaux GPU demandés par le sujet.')
else:
    # Vérifier les opérations de détection CUDA avant un long entraînement.
    torchvision.ops.nms(torch.tensor([[0., 0., 2., 2.]], device='cuda'),
                        torch.tensor([.9], device='cuda'), .5)
    print('Expérience complète : 54 000 train, 6 000 validation, 10 000 test.')


def show_result(result):
    folder = ROOT / 'results' / PROFILE / result['model']
    print(result['model'], '|', result['purpose'])
    display(pd.DataFrame([{
        'accuracy test': result['test']['accuracy'], 'F1 macro test': result['test']['f1_macro'],
        'loss propre au modèle': result['test']['loss'],
        'temps train (s)': result['training_seconds'], 'époque choisie sur validation': result['selected_epoch'],
        'paramètres entraînables': result['trainable_parameters'],
    }]))
    display(Image(filename=str(folder / 'learning_curves.png')))
    display(Image(filename=str(folder / 'confusion_matrix.png')))
    display(pd.read_csv(folder / 'classification_report.csv', index_col=0))
'''


EDA = [md('''
## Comprendre les données et le protocole commun

MNIST contient 60 000 images d'apprentissage et 10 000 images de test, en niveaux
 de gris, de taille 28×28. Le train officiel est séparé en **54 000 train et 6 000
validation**, de manière stratifiée, avec la graine 42. Les cinq modèles utilisent
les mêmes indices. Le test officiel reste réservé à l'évaluation finale.

Le miroir GitHub fournit les fichiers IDX MNIST standard. Les empreintes MD5
attendues par torchvision sont conservées ; le téléchargement ne dépend pas d'une
clé Kaggle. La source demandée par le sujet reste indiquée dans le README.

L'augmentation géométrique du profil complet agit uniquement sur le train :
rotation jusqu'à 10° et translation jusqu'à 5 %. Les sondes de train, la validation
et le test ne sont pas augmentés. Les statistiques de normalisation sont des
constantes MNIST/ImageNet fixées, pas des valeurs estimées sur le test.
'''), code('''
explore()
display(pd.read_csv(ROOT / 'results/eda/class_counts.csv', index_col=0))
display(Image(filename=str(ROOT / 'results/eda/examples_boxes.png')))
display(Image(filename=str(ROOT / 'results/eda/class_counts.png')))
''')]


END = md('''
## Lire les résultats sans tirer une conclusion trop vite

- L'accuracy mesure la proportion de chiffres corrects ; le F1 macro donne le même
  poids à chaque chiffre. Le rapport détaille précision, rappel, F1 et support.
- Les courbes d'accuracy du train utilisent une **sonde fixe** de 1 000 images dans
  le profil complet, ou 40 dans le profil smoke. La loss train utilise tous les lots.
- Le checkpoint est choisi au meilleur F1 macro de validation. Le test n'est calculé
  qu'après ce choix ; ses scores n'influencent ni les gradients ni l'époque retenue.
- Faster R-CNN additionne plusieurs pertes ; sa loss n'est pas directement
  comparable à l'entropie croisée des classifieurs.
- Les budgets d'époques, les tailles d'entrée et les paramètres entraînables
  diffèrent : le temps mesuré est descriptif, pas un classement à budget égal.
- Un score smoke de CPU ne permet aucune conclusion sur la supériorité d'un modèle.
  La conclusion finale devra être fondée sur les cinq exécutions **full GPU**.

### Conserver les résultats Colab sur GitHub

Dans Colab : **Fichier → Enregistrer une copie dans GitHub**, en conservant les
sorties des cellules. Le second notebook permet aussi d'exporter les CSV, JSON et
figures dans une archive à télécharger. Les poids et les clés d'accès ne sont pas
inclus dans cette archive. Ne publier que des sorties réellement exécutées.
''')


def first():
    cells = [md('''
# Atelier 2 — Partie 1 : CNN, Faster R-CNN et transfert

**Auteur : KARMAS Hammam**

Ce notebook explique et exécute les quatre modèles de la première partie.
Les sorties enregistrées dans le dépôt proviennent d'une vérification **CPU smoke**.
Dans Colab, sélectionner **Exécution → Modifier le type d'exécution → GPU**, puis
**Tout exécuter** : le profil complet GPU est sélectionné automatiquement.

La première exécution télécharge MNIST et les poids officiels (environ 1 Go pour
les trois modèles préentraînés). Les entraînements complets peuvent être longs,
particulièrement VGG16 et Faster R-CNN. Les résultats terminés sont réutilisés si
leur configuration, leur code et leurs partitions sont identiques.
'''), code(SETUP)] + EDA
    cells += [md('''
## 1. Construire et entraîner le CNN

Une convolution apprend des motifs locaux : traits, courbes ou intersections.
Le noyau 3×3, le padding 1 et le stride 1 conservent la taille spatiale. Chaque
max-pooling 2×2 divise la résolution par deux : 28→14→7.

Architecture : Conv(1,32) → ReLU → Pool → Conv(32,64) → ReLU → Pool →
Linear(3136,128) → ReLU → Dropout(0,25) → Linear(128,10).
La sortie est un ensemble de dix logits, transmis directement à CrossEntropyLoss.
L'optimiseur est AdamW, avec un taux initial de 0,001 et une pénalité de 0,0001.
'''), code('''
from src.models import CNN
display(CNN())
print(configuration('cnn', PROFILE))
cnn_result = run('cnn', profile=PROFILE, device_name=DEVICE)
show_result(cnn_result)
'''), md('''
## 2. Adapter Faster R-CNN à MNIST

Faster R-CNN détecte des objets : le backbone ResNet50/FPN extrait des cartes,
le RPN propose des régions, puis la tête RoI prédit une boîte et sa classe.
MNIST ne fournit pas de boîtes. Nous encadrons donc tous les pixels non nuls de
chaque chiffre ; après augmentation, la boîte est recalculée.

Les images deviennent RGB et le détecteur les redimensionne à 64×64. Les ancres
sont adaptées aux petits chiffres. Le modèle utilise des poids COCO officiels,
puis sa tête est remplacée : **11 classes = fond + dix chiffres**. Le chiffre zéro
est encodé par 1 pendant la détection ; le fond reste 0.

Pour la comparaison par image, nous retenons la détection la plus confiante et
reconvertissons son étiquette en chiffre. Aucune détection donne -1 et compte
comme une erreur. Nous conservons aussi l'IoU de la boîte et le taux de réussite
« classe correcte et IoU ≥ 0,5 ». Ce protocole n'est pas une mesure mAP.
'''), code('''
print(configuration('fasterrcnn', PROFILE))
faster_result = run('fasterrcnn', profile=PROFILE, device_name=DEVICE)
show_result(faster_result)
display(pd.DataFrame([{key: faster_result['test'][key] for key in
                      ['no_detection_rate', 'mean_iou', 'label_and_iou50_accuracy']}]))
'''), md('''
## 3. Fine-tuning de VGG16

Les filtres ImageNet constituent un point de départ. Les images sont redimensionnées
à 224×224, répétées sur trois canaux et normalisées avec les constantes ImageNet.
Nous remplaçons la dernière couche par dix sorties. Le dernier bloc convolutif
et cette nouvelle tête sont entraînés ; les autres poids restent figés.
Il s'agit donc d'un fine-tuning partiel, et pas seulement d'une tête entraînée.
Le taux d'apprentissage est plus petit : 0,0001.
'''), code('''
print(configuration('vgg16', PROFILE))
vgg_result = run('vgg16', profile=PROFILE, device_name=DEVICE)
show_result(vgg_result)
'''), md('''
## 4. Fine-tuning d'AlexNet

La préparation des images est identique à VGG16. Le dernier bloc convolutif et
la tête à dix sorties sont ajustés. AlexNet et VGG16 disposent de poids ImageNet
préentraînés, alors que le CNN et le ViT partent de poids aléatoires : cette
connaissance préalable doit être mentionnée lors de l'interprétation.
'''), code('''
print(configuration('alexnet', PROFILE))
alex_result = run('alexnet', profile=PROFILE, device_name=DEVICE)
show_result(alex_result)
'''), md('''
## 5. Comparer les quatre modèles de cette partie

Le tableau ci-dessous décrit le profil affiché au début du notebook. Dans le
profil smoke, les scores sont uniquement des preuves de fonctionnement du code.
Les temps sont les boucles d'entraînement synchronisées ; ils excluent les
 téléchargements, la validation, le test et la sauvegarde des poids.
'''), code('''
rows = []
for result in [cnn_result, faster_result, vgg_result, alex_result]:
    rows.append({'modèle': result['model'], 'profil': result['profile'],
                 'accuracy': result['test']['accuracy'], 'F1 macro': result['test']['f1_macro'],
                 'loss propre au modèle': result['test']['loss'],
                 'temps train (s)': result['training_seconds'],
                 'époques': result['config']['epochs'], 'matériel': result['device_name']})
display(pd.DataFrame(rows))
print('Continuer avec 02_vit_comparaison.ipynb pour le ViT et le tableau des cinq modèles.')
'''), END]
    return cells


def second():
    return [md('''
# Atelier 2 — Partie 2 : Vision Transformer depuis zéro

**Auteur : KARMAS Hammam**

Ce notebook implémente la logique du ViT puis le compare aux modèles de la partie 1.
Il réutilise le même protocole. Les sorties du dépôt sont des vérifications CPU ;
le profil complet exige un GPU Colab. Exécuter les deux notebooks pour obtenir
les cinq résultats complets, ou utiliser la commande CLI indiquée plus bas.

Le tutoriel du sujet sert de référence pédagogique. L'implémentation locale est
originale : aucun backbone ViT préentraîné, ni timm, ni TransformerEncoder tout fait.
'''), code(SETUP)] + EDA + [md('''
## 1. Transformer une image en séquence

Une image 28×28 est divisée en **16 patches de 7×7**. Chaque patch contient
49 valeurs, projetées par une couche linéaire vers 64 dimensions. Un token
apprenable **CLS** résume l'image ; une position apprise indique l'emplacement
des patches. La séquence comporte donc 17 tokens.

Trois blocs encodeurs appliquent une normalisation, une attention à quatre têtes,
une connexion résiduelle et un MLP avec GELU. L'attention est calculée explicitement :
**softmax(QKᵀ / √d) V**. Les têtes regardent différentes relations entre patches.
La représentation finale du token CLS alimente une tête à dix logits.
'''), code('''
from src.models import ViT
model_example = ViT()
display(model_example)
print('Forme des positions :', tuple(model_example.position.shape))
print('Sortie sur deux images :', tuple(model_example(torch.randn(2, 1, 28, 28)).shape))
del model_example
'''), md('''
## 2. Entraîner le ViT

Le profil complet utilise AdamW, un taux de 0,0003, weight_decay=0,01,
12 époques et des lots de 128. Le Dropout vaut 0,1 dans les blocs.
Le meilleur checkpoint est sélectionné au F1 macro de validation.

Un ViT n'a pas le même biais local qu'un CNN. Il peut nécessiter davantage de
données ou de réglages. Cela constitue une hypothèse d'interprétation ; les
résultats complets détermineront ce qui est observé dans cette expérience.
'''), code('''
print(configuration('vit', PROFILE))
vit_result = run('vit', profile=PROFILE, device_name=DEVICE)
show_result(vit_result)
'''), md('''
## 3. Finaliser les cinq expériences dans cette session

Si les résultats de la première partie ne sont pas présents dans cette session
Colab, la cellule suivante entraîne les modèles manquants. Les résultats déjà
terminés et compatibles sont réutilisés. Le tableau final n'est créé que si les
cinq modèles ont terminé sur les mêmes partitions et le même profil.
'''), code('''
from src.config import MODELS
for name in MODELS:
    run(name, profile=PROFILE, device_name=DEVICE)
comparison = compare(PROFILE)
display(comparison)
display(Image(filename=str(ROOT / 'results' / PROFILE / 'comparison.png')))
'''), md('''
## 4. Interpréter la comparaison

Comparer accuracy et F1 macro sur les mêmes images, puis regarder les confusions
par chiffre. Relier les différences aux paramètres entraînables, au préentraînement,
aux tailles d'entrée et aux budgets d'époques. Les losses sont diagnostiques :
la somme des pertes de détection ne mesure pas la même chose qu'une entropie croisée.

La comparaison des temps n'est pertinente qu'en tenant compte du matériel.
Les époques sont volontairement différentes entre les architectures ; le tableau
ne constitue pas un benchmark à temps ou à nombre de mises à jour égaux.

**La conclusion académique finale reste à compléter avec les résultats GPU.**
Les sorties CPU affichées ici ne permettent pas de déclarer un modèle supérieur.
'''), code('''
from scripts.export_results import export
archive = export(PROFILE)
print('Archive à conserver :', archive.name)
try:
    from google.colab import files
    files.download(str(archive))
except ImportError:
    print('Exécution locale : archive disponible à la racine du dépôt.')
'''), md('''
## 5. Synthèse des notions abordées

Cet atelier montre comment une convolution exploite les motifs locaux, comment
le transfert réutilise des caractéristiques apprises et comment un ViT relie des
patches par attention. L'adaptation de Faster R-CNN oblige à distinguer classification
et détection, ainsi qu'accuracy par image et qualité de localisation.

La synthèse personnelle finale doit décrire ce qui a été compris et les résultats
réellement obtenus sur GPU. Aucun résultat Colab n'est supposé à partir du profil CPU.
'''), END]


if __name__ == '__main__':
    target = ROOT / 'notebooks'
    target.mkdir(exist_ok=True)
    for filename, cells in [('01_cnn_detection_transfert.ipynb', first()),
                             ('02_vit_comparaison.ipynb', second())]:
        notebook = nbf.v4.new_notebook(cells=cells, metadata={
            'kernelspec': {'display_name': 'Python 3', 'name': 'python3', 'language': 'python'},
            'language_info': {'name': 'python', 'version': '3.12.14'},
            'colab': {'name': filename, 'provenance': []}})
        nbf.write(notebook, target / filename)
