from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Initialisation de l'application Flask
app = Flask(__name__)

# Chargement du modèle Keras
# Assurez-vous que le fichier 'mnist_model.h5' est présent dans le même dossier
model = keras.models.load_model('mnist_model.h5')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)

    # Vérification de la présence des données
    if 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400

    try:
        # Conversion en tableau numpy
        image_data = np.array(data['image'])
        # Vérification de la taille
        if image_data.size != 784:
            return jsonify({'error': 'Image must be flattened (784 values)'}), 400

        # Mise en forme et normalisation
        image_data = image_data.reshape(1, 784).astype("float32") / 255.0

        # Prédiction avec le modèle
        prediction = model.predict(image_data)
        predicted_class = int(np.argmax(prediction, axis=1)[0])

        # Réponse JSON
        return jsonify({
            'prediction': predicted_class,
            'probabilities': prediction.tolist()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Lancement du serveur Flask
    app.run(host='0.0.0.0', port=5000)
