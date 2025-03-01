from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('digit_recognizer_model.h5')

# Route to serve the frontend
@app.route('/')
def index():
    return render_template('index.html')

# Define a route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Check if the image is provided in the POST request
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image = request.files['image']

    # Convert the image into a format that the model can process
    image = Image.open(io.BytesIO(image.read()))
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image = np.array(image)
    image = image / 255.0  # Normalize pixel values to [0,1]
    image = image.reshape(1, 784)  # Flatten to 1D array (28*28 pixels)

    # Predict the digit
    prediction = model.predict(image)
    predicted_label = np.argmax(prediction, axis=1)[0]

    return jsonify({'predicted_label': int(predicted_label)})

# Start the Flask application
if __name__ == '__main__':
    app.run(debug=True)
