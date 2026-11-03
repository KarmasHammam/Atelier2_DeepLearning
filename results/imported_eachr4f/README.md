# Résultats de référence importés de Eachr4f

Source : https://github.com/Eachr4f/atelier2-mnist
Commit : `47654402f475fbe0e1d608b8e8417f28bb080d27`.

Les métriques de `comparison.csv` et `metrics.json` sont extraites des sorties
conservées dans `references/eachr4f/notebooks/`. Elles n'ont pas été recalculées
à partir de checkpoints et ne constituent pas une nouvelle exécution indépendante.

| Modèle | Accuracy | F1 macro | Loss test affichée | Durée affichée |
| --- | ---: | ---: | ---: | --- |
| CNN | 99,56 % | 0,9956 | 0,0125 | 339,7 s |
| Faster R-CNN | 99,22 % | 0,9921 | Non rapportée | 79,9 min |
| VGG16 | 99,49 % | 0,9949 | 0,0226 | 6,0 min |
| AlexNet | 99,25 % | 0,9924 | 0,0236 | 3,8 min |
| ViT | 87,92 % | 0,8772 | 0,3746 | 62,6 s |

Limites : partitions et budgets différents ; un ou deux GPU selon le modèle ;
validation Faster R-CNN en mode train qui modifie les BatchNorm ; absence de détection
assimilée au chiffre zéro ; ViT sans validation indépendante et loss finale moyennée
par lot plutôt que par image. Le README principal explique les corrections.

Ces scores ne valident pas les versions de `notebooks/colab/`, dont le protocole a changé.
