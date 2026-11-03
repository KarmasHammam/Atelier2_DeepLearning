# État de l'atelier 2

- Implémentation des cinq modèles : présente.
- MNIST : téléchargé et vérifié par les MD5 de torchvision.
- Poids VGG16/AlexNet/Faster R-CNN : officiels, réellement chargés.
- Profil CPU `smoke` : cinq exécutions réelles, une époque, 64/40/80 images.
- Tests : sept contrôles réussis.
- GPU dans l'environnement de préparation : absent (`torch.cuda.is_available() == False`).
- Profil complet `full` : **non exécuté ici** ; aucune métrique GPU n'est revendiquée.

Utiliser les notebooks Colab avec un GPU pour produire les cinq expériences complètes.
Les résultats CPU servent à vérifier le code ; ils ne permettent pas un classement
scientifique des architectures. Le README décrit la procédure d'export.
