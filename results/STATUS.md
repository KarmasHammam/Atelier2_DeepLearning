# État des résultats

1. **Référence importée** : cinq notebooks exécutés de Eachr4f, conservés inchangés
   dans `references/eachr4f/notebooks/`. Métriques dans `results/imported_eachr4f/`.
   Leur origine est attribuée ; aucune reproduction GPU indépendante n'est revendiquée.
2. **Adaptation corrigée** : cinq notebooks dans `notebooks/colab/`, à réexécuter sur
   GPU. Aucune sortie d'origine n'est présentée comme résultat du code modifié.
   Les nouvelles mesures iront dans `results/adapted_gpu/`.
3. **Protocole initial du dépôt** : résultats CPU limités conservés dans `results/smoke/`.
   Les notebooks historiques et les enregistrements Colab déjà présents sont préservés.

Treize tests locaux passent. Les modèles CNN/ViT adaptés ont été vérifiés sur CPU,
les scripts DDP compilent et les défauts d'évaluation identifiés ont été corrigés.
Les entraînements adaptés complets et les chemins NCCL/CUDA restent à valider sur GPU.
