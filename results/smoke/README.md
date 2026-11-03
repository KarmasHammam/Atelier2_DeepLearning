# SMOKE CPU — 64 images de train, 1 époque : ne pas conclure sur les architectures

Les losses Faster R-CNN et les entropies croisées ont des définitions différentes : ne pas les classer ensemble.
Les temps excluent téléchargement, validation, test et écriture des poids. Les budgets et tailles d’entrée diffèrent.
Même matériel déclaré pour les cinq exécutions : True.

| Modèle | Accuracy | F1 macro | Loss propre au modèle | Train (s) | Époques |
| --- | --- | --- | --- | --- | --- |
| cnn | 0.3625 | 0.2764 | 2.2467 | 0.13 | 1 |
| fasterrcnn | 0.2625 | 0.1676 | 0.2838 | 26.01 | 1 |
| vgg16 | 0.2625 | 0.2159 | 2.1607 | 14.00 | 1 |
| alexnet | 0.2750 | 0.2277 | 2.1263 | 1.07 | 1 |
| vit | 0.2000 | 0.1138 | 2.2591 | 0.11 | 1 |
