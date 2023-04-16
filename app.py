from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from skimage.transform import resize
import numpy as np
import io
from PIL import Image
from flask import Flask, request, send_file
from flask_cors import CORS

app = Flask(__name__)

IMG_HEIGHT = 720
IMG_WIDTH = 1280
IMG_CHANNELS = 3

model_path = 'model.h5'
loaded_model = load_model(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    # Load and preprocess the image
    image = request.files['image']
    input_img = Image.open(image)
    input_img_array = img_to_array(input_img)
    input_img_resized = resize(input_img_array, (IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), mode='constant', preserve_range=True)
    input_img_normalized = input_img_resized / 255.0
    input_batch = np.expand_dims(input_img_normalized, axis=0)

    # Make a prediction using the loaded model
    predictions = loaded_model.predict(input_batch)
    predicted_mask = predictions[0]

    # Convert the predicted mask to a PIL Image
    predicted_mask = predicted_mask.squeeze() * 255
    if predicted_mask.ndim == 2:
        im = Image.fromarray(predicted_mask.astype(np.uint8), mode='L')
    else:
        im = Image.fromarray(predicted_mask.astype(np.uint8), mode='RGB')

    # Save the PIL Image to a buffer
    output_buffer = io.BytesIO()
    im.save(output_buffer, format='PNG')
    output_buffer.seek(0)

    # Return the predicted mask as a file
    return send_file(output_buffer, mimetype='image/png')

if __name__ == '__main__':
    CORS(app)
    app.run(debug=True)
