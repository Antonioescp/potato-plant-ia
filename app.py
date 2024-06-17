import base64
import tensorflow as tf
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from io import BytesIO
from PIL import Image
from keras.layers import Rescaling
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = tf.keras.models.load_model('models/best_model.h5')

scaler = Rescaling(1. / 255)

# Function to preprocess the image for the model
def preprocess_image(image):
    image = image.resize((256, 256))  # Resize to the input size required by the model

    if image.mode != "RGB":
        image = image.convert("RGB")

    image = np.array(image)
    image = scaler(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to classify the image
def classify_image(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    class_index = np.argmax(predictions, axis=1)[0]
    class_names = ["Saludable", "Tizón temprano", "Tizón tardío"]
    class_name = class_names[class_index]
    probability = predictions[0, class_index]  # Get probability for the predicted class
    percent = int(round(probability, 2) * 100)

    # Map probability to string based on thresholds
    if probability > 0.97:
        prob_str = "Alta"
    elif probability > 0.80:
        prob_str = "Media"
    else:
        prob_str = "Baja"

    return class_name, image_to_base64(image), prob_str, percent

# Function to convert image to base64
def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = "data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")
    return img_str

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        files = request.files.getlist('files[]')
        results = []

        for file in files:
            if file:
                filename = secure_filename(file.filename)
                image = Image.open(BytesIO(file.read()))  # Read image into memory
                classification, img_str, probability, percent = classify_image(image)
                results.append((filename, img_str, classification, probability, percent))

        return render_template('result.html', results=results)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
