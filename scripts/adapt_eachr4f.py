"""Adapter les instantanés attribués de Eachr4f en notebooks GPU portables.

Les originaux restent intacts dans references/eachr4f/. Les versions corrigées
n'héritent d'aucune sortie exécutée : elles doivent être relancées sur GPU.
"""
from pathlib import Path
import hashlib
import json
import re
import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'references/eachr4f/notebooks'
COMMIT = '47654402f475fbe0e1d608b8e8417f28bb080d27'
FILES = {
    'cnn': ('MNIST_CNN_Result.ipynb', '01_cnn.ipynb'),
    'fasterrcnn': ('Faster_RCNN_MNIST.ipynb', '02_faster_rcnn.ipynb'),
    'vgg16': ('VGG16_FineTuning_MNIST.ipynb', '03_vgg16.ipynb'),
    'alexnet': ('AlexNet_FineTuning_MNIST.ipynb', '04_alexnet.ipynb'),
    'vit': ('ViT_MNIST.ipynb', '05_vit.ipynb'),
}

BOOTSTRAP = '''
from pathlib import Path
import os, sys, subprocess, importlib.util
ROOT = Path.cwd()
if not (ROOT / 'src/adaptation.py').exists():
    # Accepter un lancement depuis notebooks/colab/ ou depuis Colab lui-même.
    for candidate in list(ROOT.parents):
        if (candidate / 'src/adaptation.py').exists():
            ROOT = candidate
            break
    else:
        ROOT = Path('/content/Atelier2_DeepLearning') if Path('/content').exists() else Path('/kaggle/working/Atelier2_DeepLearning')
        if not ROOT.exists():
            subprocess.run(['git', 'clone', 'https://github.com/KarmasHammam/Atelier2_DeepLearning.git', str(ROOT)], check=True)
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
os.environ['PYTHONPATH'] = str(ROOT) + os.pathsep + os.environ.get('PYTHONPATH', '')
if importlib.util.find_spec('google') and importlib.util.find_spec('google.colab'):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements-colab.txt'])
import torch
if not torch.cuda.is_available():
    raise RuntimeError('Active un GPU : Colab > Exécution > Modifier le type d’exécution > GPU.')
from src.adaptation import prepare_work, save_result
WORK_DIR = prepare_work('{model}')
os.environ['ATELIER2_WORK_DIR'] = str(WORK_DIR)
print('Adaptation KARMAS Hammam ; source Eachr4f, commit {commit}')
print('Matériel de cette session :', torch.cuda.get_device_name(0))
print('Les résultats de référence importés ne sont pas ceux de cette nouvelle exécution.')
'''


def paths(text):
    # Convertir les chemins littéraux, y compris dans le script DDP embarqué.
    pattern = r"(['\"])/kaggle/working([^'\"\n]*)\1"
    return re.sub(pattern, lambda m: "str(WORK_DIR / " + repr(m.group(2).lstrip('/')) + ")", text)


def adapt_worker(text, model):
    text = paths(text)
    text = 'import os\nfrom pathlib import Path\nWORK_DIR=Path(os.environ["ATELIER2_WORK_DIR"])\n' + text
    if model == 'fasterrcnn':
        text = 'from src.engine import detection_loss\n' + text
        old = "losses=model.module(images,move_targets(targets,device))\n                    vtotal+=sum(losses.values()).item()*len(images); vn+=len(images)"
        new = "value=detection_loss(model.module,images,move_targets(targets,device),device)\n                    vtotal+=value*len(images); vn+=len(images)"
        assert old in text
        text = text.replace(old, new)
    insertion = '''    if rank==0:
        torch.cuda.synchronize()
        with open(str(WORK_DIR / 'timing.json'), 'w') as timing_file:
            json.dump({'training_seconds': time.time()-start}, timing_file)
        with open'''
    text, count = re.subn(r'(?m)^    if rank==0:\n        with open', lambda m: insertion, text)
    assert count == 1, (model, count)
    return text


def modify(text, model, index):
    if model != 'vit' and index == 3:
        return "# Les IDX MNIST ont été téléchargés et vérifiés par prepare_work.\nprint('Données locales :', WORK_DIR / 'MNIST/raw')"
    # Les fichiers de la source exigent deux T4 ; torchrun gère aussi un processus.
    text = re.sub(r'if NUM_GPUS\s*<\s*2:\s*raise RuntimeError\([^\n]*\)',
                  'if NUM_GPUS < 1: raise RuntimeError("Un GPU au moins est nécessaire.")', text)
    text = re.sub(r'if NUM_GPUS\s*<\s*2:\n[ \t]+print\([^\n]*Attention[^\n]*\)',
                  'print("DDP utilisera un ou deux GPU selon cette session.")', text)
    text = re.sub(r'(?m)^.*print\([^\n]*(?:/kaggle/input|Attention : ce notebook continuera)[^\n]*\)\s*$', '', text)
    text = re.sub(r'TRAIN_SIZE\s*=\s*59000', 'TRAIN_SIZE = 54000', text)
    text = re.sub(r'VAL_SIZE\s*=\s*1000', 'VAL_SIZE = 6000', text)
    text = text.replace("'--nproc_per_node=2'", "f'--nproc_per_node={min(NUM_GPUS, 2)}'")
    text = text.replace('"--nproc_per_node=2"', 'f"--nproc_per_node={min(NUM_GPUS, 2)}"')
    text = text.replace("print('Lancement DDP sur les 2 GPU T4 ...')", "print('Lancement DDP :', min(NUM_GPUS, 2), 'GPU')")
    text = text.replace('training_seconds=None  # durée détaillée affichée par le script DDP',
                        "training_seconds=json.loads((WORK_DIR / 'timing.json').read_text())['training_seconds']")
    if 'worker_script =' in text:
        pattern = r'(worker_script\s*=\s*r?""")(.*?)(""")'
        text, count = re.subn(pattern, lambda m: m.group(1) + adapt_worker(m.group(2), model) + m.group(3), text, flags=re.S)
        assert count == 1, model
    text = paths(text)
    # La validation commune est un tirage randperm seed=42 ; mêmes 54k/6k partout.
    if model == 'vit':
        text = text.replace('np.random.seed(0)', 'np.random.seed(42)').replace('torch.manual_seed(0)', 'torch.manual_seed(42)')
        if index == 14:
            return '''# Même partage 54 000 / 6 000 que les quatre autres modèles.
transform = ToTensor()
full_train_set = MNIST(root=str(WORK_DIR), train=True, download=False, transform=transform)
test_set = MNIST(root=str(WORK_DIR), train=False, download=False, transform=transform)
indices = torch.randperm(60000, generator=torch.Generator().manual_seed(42)).tolist()
train_set = torch.utils.data.Subset(full_train_set, indices[6000:])
val_set = torch.utils.data.Subset(full_train_set, indices[:6000])
train_loader = DataLoader(train_set, shuffle=True, batch_size=128, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_set, shuffle=False, batch_size=128, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_set, shuffle=False, batch_size=128, num_workers=2, pin_memory=True)
print(len(train_set), len(val_set), len(test_set))'''
        if index == 18:
            text = text.replace('test_losses', 'val_losses').replace('test_epoch_loss', 'val_epoch_loss')
            text = text.replace('evaluate_loss(test_loader)', 'evaluate_loss(val_loader)').replace('test loss:', 'val loss:')
            text = text.replace('training_start_time = time.time()',
                                'best_val_loss = float("inf")\nbest_state = None\ntorch.cuda.synchronize()\ntraining_start_time = time.time()')
            text = text.replace('    val_losses.append(val_epoch_loss)', '''    val_losses.append(val_epoch_loss)
    if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}''')
            text = text.replace('training_seconds = time.time()', 'torch.cuda.synchronize()\ntraining_seconds = time.time()')
            text += '\n# Restaurer le checkpoint choisi sur validation, jamais sur le test.\nmodel.load_state_dict(best_state)\n'
        if index == 20:
            text = text.replace('test_losses', 'val_losses').replace('label="Test loss"', 'label="Validation loss"')
        if index == 22:
            text = text.replace('with torch.no_grad():', 'model.eval()\nwith torch.no_grad():', 1)
            text = text.replace('test_loss += loss.detach().cpu().item() / len(test_loader)',
                                'test_loss += loss.detach().cpu().item() * len(y)')
            text = text.replace('    test_f1_macro =', '    test_loss /= total\n    test_f1_macro =')
    if model == 'fasterrcnn' and index == 13:
        text = '''# Prédiction sans les boîtes vraies ; loss diagnostique avec BatchNorm figées.
from sklearn.metrics import accuracy_score, f1_score, classification_report
from src.engine import detection_loss, image_prediction
model.eval()
y_true, y_pred = [], []
loss_sum, number = 0., 0
with torch.no_grad():
    for images, targets, digits in test_loader:
        images = [image.to(device) for image in images]
        moved = [{key: value.to(device) for key, value in target.items()} for target in targets]
        outputs = model(images)
        for output, digit in zip(outputs, digits):
            predicted, confidence, box = image_prediction(output)
            y_pred.append(predicted)  # -1 si aucune détection : jamais assimilé au chiffre zéro.
            y_true.append(int(digit))
        loss_sum += detection_loss(model, images, moved, device) * len(images)
        number += len(images)
test_loss = loss_sum / number
print('Accuracy :', accuracy_score(y_true, y_pred))
print('F1 macro :', f1_score(y_true, y_pred, labels=list(range(10)), average='macro', zero_division=0))
print('Loss diagnostique détection :', test_loss, '(non comparable à une cross-entropy seule)')
print(classification_report(y_true, y_pred, labels=list(range(10)), digits=4, zero_division=0))'''
    return text


def main():
    target = ROOT / 'notebooks/colab'
    target.mkdir(exist_ok=True)
    hashes = {}
    for model, (original, destination) in FILES.items():
        source = SOURCE / original
        hashes[original] = hashlib.sha256(source.read_bytes()).hexdigest()
        notebook = nbf.read(source, as_version=4)
        header = f'''# Atelier 2 — {model.upper()} — adaptation KARMAS Hammam

Source : [Eachr4f/atelier2-mnist](https://github.com/Eachr4f/atelier2-mnist),
commit `{COMMIT}`. Les originaux exécutés sont conservés dans
[`references/eachr4f/notebooks/`](../../references/eachr4f/notebooks/).

**Cette version est corrigée et n'a pas encore été exécutée sur GPU.** Ses sorties
ont été effacées pour ne pas attribuer les anciens scores au code modifié.
Dans Colab : activer un GPU puis **Tout exécuter**. Un ou deux GPU sont acceptés.

Adaptations : chemins portables et téléchargement vérifié ; partage commun
54 000 train / 6 000 validation / 10 000 test (seed 42) ; export des nouveaux résultats.
Faster R-CNN : BatchNorm figées en validation, absence de détection = -1 et loss test
explicitement diagnostique. ViT : vraie validation, checkpoint sélectionné sur
validation et moyenne de loss test pondérée par les effectifs des lots.

Les architectures et réglages de base proviennent du dépôt attribué, avec une
convolution CNN à trois blocs, des têtes compactes pour VGG16/AlexNet et le ViT
pédagogique à 8 dimensions. Le partage et les corrections changent le protocole :
**ne pas réutiliser les scores d'origine comme résultats de cette adaptation.**'''
        cells = [nbf.v4.new_markdown_cell(header), nbf.v4.new_code_cell(BOOTSTRAP.format(model=model, commit=COMMIT))]
        for index, cell in enumerate(notebook.cells):
            if cell.cell_type == 'code':
                cell.source = modify(cell.source, model, index)
                cell.outputs, cell.execution_count, cell.metadata = [], None, {}
            else:
                text = cell.source
                text = text.replace('59 000', '54 000').replace('1 000', '6 000')
                text = text.replace('sur deux GPU', 'sur un ou deux GPU')
                text = re.sub(r'Dans Kaggle, ajoutez[^\n]*',
                              'Activer un GPU dans Colab puis exécuter les cellules dans l’ordre. Le téléchargement MNIST est automatique.', text)
                text = text.replace('Kaggle T4 x2', 'Colab/Kaggle GPU')
                if model == 'vit' and index == 17:
                    text = '''## 9. Entraînement et suivi de la validation

Le réseau apprend sur les 54 000 images du train. La loss de validation est
calculée après chaque époque sur 6 000 autres images. Les meilleurs poids sont
sélectionnés selon cette loss, puis restaurés avant l'évaluation finale du test.'''
                if model == 'vit' and index == 19:
                    text = '''## 10. Courbes train et validation

Le temps mesure entraînement, validation et sélection des poids. Le test ne
participe pas à ces courbes ; il reste réservé à l'évaluation finale.'''
                if model == 'vit' and text.startswith('## Conclusion'):
                    text = '''## Interprétation après cette nouvelle exécution

Comparer la diminution de la loss train et celle de validation, puis analyser
l'accuracy et le F1 du test. Une divergence train/validation peut signaler du
surapprentissage. Les limites du modèle compact doivent être discutées sans
conclure à la supériorité générale d'une famille de modèles.

Les 87,92 % d'accuracy et le F1 de 0,8772 appartiennent à l'exécution d'origine,
conservée dans references/eachr4f/. Cette version utilise un autre partage et une
sélection sur validation : ses propres résultats seront ceux affichés après
exécution, et non ces valeurs de référence.'''
                cell.source = text
            cells.append(cell)
        truths = {'cnn': ('all_labels', 'all_preds'), 'fasterrcnn': ('y_true', 'y_pred'),
                  'vgg16': ('y_true', 'y_pred'), 'alexnet': ('y_true', 'y_pred'),
                  'vit': ('all_targets', 'all_predictions')}
        truth, predictions = truths[model]
        # VGG/AlexNet mesurent leur durée dans le processus DDP, pas dans le parent.
        seconds = 'None' if model in ('vgg16', 'alexnet') else 'training_seconds'
        cells += [nbf.v4.new_markdown_cell('''## Enregistrer cette nouvelle exécution

Les métriques sont calculées depuis les prédictions de cette session GPU. Le dossier
`results/adapted_gpu/` reste distinct des résultats importés et des anciens profils.
Enregistrer aussi le notebook dans GitHub **avec ses sorties**. L'archive permet de
conserver les résultats, même si les cinq modèles ne sont pas tous terminés.'''),
                  nbf.v4.new_code_cell(f'''result = save_result('{model}', {truth}, {predictions}, test_loss, {seconds}, WORK_DIR)
from scripts.export_adapted_results import export
archive = export(allow_partial=True)
print('Archive :', archive)
try:
    from google.colab import files
    files.download(str(archive))
except ImportError:
    print('Archive disponible localement.')''')]
        notebook.cells = cells
        notebook.metadata = {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
                             'language_info': {'name': 'python'}, 'adaptation': {'author': 'KARMAS Hammam',
                             'source': 'https://github.com/Eachr4f/atelier2-mnist', 'commit': COMMIT,
                             'gpu_executed': False}}
        nbf.validate(notebook)
        nbf.write(notebook, target / destination)
    (ROOT / 'references/eachr4f/manifest.json').write_text(json.dumps({
        'repository': 'https://github.com/Eachr4f/atelier2-mnist', 'commit': COMMIT,
        'original_files_sha256': hashes, 'original_notebooks_preserved_without_modification': True,
        'adaptation_author': 'KARMAS Hammam', 'adapted_notebooks_gpu_executed': False}, indent=2) + '\n')


if __name__ == '__main__':
    main()
