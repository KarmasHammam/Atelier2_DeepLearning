# Provenance de l'import

- Dépôt source : https://github.com/Eachr4f/atelier2-mnist
- Commit importé : `47654402f475fbe0e1d608b8e8417f28bb080d27`.
- Les cinq fichiers dans `notebooks/` sont des copies **inchangées**, avec leurs
  cellules et sorties enregistrées. `README_source.md` conserve le README source.
- `manifest.json` donne les empreintes SHA-256 des notebooks d'origine.
- Les métriques de `../../results/imported_eachr4f/` sont lues dans les sorties source,
  sans réentraîner les modèles. Le README source annonce à tort l'absence de F1 ViT :
  la sortie du notebook contient **0,8772**.

L'adaptation de **KARMAS Hammam** se trouve dans `../../notebooks/colab/`.
Elle conserve l'attribution à Eachr4f. Les sorties des versions corrigées sont effacées :
les anciens scores ne constituent pas une validation du code modifié.

## Différences qui nécessitent de nouvelles mesures

- Chemins portables et MNIST téléchargé avec vérification des MD5.
- Un ou deux GPU, au lieu d'exiger systématiquement deux T4.
- Partitions harmonisées : 54 000 / 6 000 / 10 000 et graine 42.
- Faster R-CNN : BatchNorm figées pour la loss de validation ; absence de détection
  comptée comme -1 ; loss test diagnostique ajoutée.
- ViT : validation séparée, meilleur checkpoint choisi sur validation et moyenne
  de la loss test pondérée par le nombre d'images de chaque lot.
- Export des nouvelles prédictions et métriques avec leur matériel et leur origine.

Les sorties historiques restent visibles pour étude. Les problèmes méthodologiques
repérés sont décrits dans le README principal ; ils ne sont pas corrigés rétroactivement
à l'intérieur des fichiers d'origine.
