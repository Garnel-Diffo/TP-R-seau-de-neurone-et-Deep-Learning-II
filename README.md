# TP1 — Exercice 1 : Entraînement d'un réseau fully-connected sur MNIST

**Portée de ce fichier README**  
Ce document couvre **uniquement** l'Exercice 1 du TP (Partie 1) : le code de formation du modèle MNIST (Listing 1). Il explique de manière chronologique et pratique comment **mettre en place le projet**, **créer l'environnement virtuel**, **installer les bibliothèques** nécessaires et **lancer** le script `train_model.py`.

---

## 1. Description du projet (exercice 1)
Ce projet entraîne un réseau de neurones fully-connected (Dense) pour la classification des chiffres manuscrits du jeu de données MNIST. Le modèle :

- Lit MNIST via `keras.datasets.mnist`.
- Normalise les images (valeurs entre 0 et 1).
- Aplatit les images en vecteurs de taille 784 (28×28) pour un réseau fully-connected.
- Utilise une architecture : `Dense(512, relu)` → `Dropout(0.2)` → `Dense(10, softmax)`.
- S'entraîne avec `optimizer='adam'`, `loss='sparse_categorical_crossentropy'`.
- Sauvegarde le modèle final sous `mnist_model.h5`.

Le code principal se trouve dans `train_model.py` (contenu donné ci-dessous).

---

## 2. Prérequis
- Python 3.8 — 3.11 recommandé.
- ~1 GB d'espace disque libre.
- Connexion internet pour télécharger MNIST et les paquets.
- (Optionnel) GPU compatible si vous souhaitez accélérer l'entraînement — installer la version GPU de TensorFlow adaptée à votre configuration.

---

## 3. Structure minimale du répertoire
Placez les fichiers suivants à la racine du projet, exemple :
```
tp1_exo1/
├─ train_model.py
└─ README.md   # ce fichier
```

---

## 4. Création de l'environnement virtuel 
Ouvrez un terminal **dans le répertoire du projet** puis exécutez :

### Linux / macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows (PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Vérifier que l'environnement est activé :  
```bash
python --version
where python   # ou `which python` sous Linux
```

---

## 5. Installer les bibliothèques requises

Installez :
```bash
pip install --upgrade pip
pip install tensorflow numpy
```

> Si vous utilisez un GPU, remplacez `tensorflow` par la version GPU appropriée compatible avec votre CUDA/cuDNN.

---

## 6. Fichier `train_model.py` (code exact à utiliser)
Créez `train_model.py` avec **exactement** le contenu ci-dessous :

```python
# train_model.py
"""
Entraînement d'un réseau fully-connected sur MNIST.
Usage : python train_model.py
Produit : mnist_model.h5
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np

# Chargement du jeu de données MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalisation [0,1] et conversion en float32
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Vectorisation (flatten) pour le modèle dense
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# Définition du modèle
model = keras.Sequential([
    keras.layers.Dense(512, activation="relu", input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10, activation="softmax")
])

# Compilation
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Entraînement avec early stopping (optionnel pour éviter le surapprentissage)
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.1
)

# Évaluation
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Précision sur les données de test: {test_acc:.4f}")

# Sauvegarde du modèle (format HDF5 ou SavedModel)
model.save("mnist_model.h5")
print("Modèle sauvegardé sous mnist_model.h5")
```

---

## 7. Lancer le script (ordre exact)
Une fois l'environnement activé et les dépendances installées, exécutez :
```bash
python train_model.py
```

Résultats attendus :
- Logs d'entraînement (loss / accuracy) pour chaque epoch.
- À la fin : impression de la précision sur le jeu de test (ex. `Précision sur les données de test: 0.98xx`).
- Fichier `mnist_model.h5` créé dans le dossier courant.

---

## 8. Vérifications simples en cas d'erreur
- Si `ModuleNotFoundError: No module named 'tensorflow'` → vérifier activation du venv et réinstaller `pip install -r requirements.txt`.
- Si erreur liée à la version Python → vérifier `python --version` et utiliser une version supportée.
- Si l'entraînement est très lent et que vous avez un GPU → installez la version GPU adaptée.

---

## MLflow
Lancer l'UI : `mlflow ui` puis exécuter `python train_model_mlflow.py`.

## Docker


## Rendu
- Lien GitHub : [(https://github.com/Garnel-Diffo/TP-R-seau-de-neurone-et-Deep-Learning-II.git)](https://github.com/Garnel-Diffo/TP-R-seau-de-neurone-et-Deep-Learning-II.git)
- Rapport PDF (Overleaf) : (coller le lien Overleaf)

