import keras
import numpy as np
from flask import Flask, request, jsonify
from utils import img_to_mat
from PIL import Image
import os

app = Flask(__name__)

PORT = os.getenv("PORT", "8080")
MODEL_PATH = os.getenv("MODEL_PATH", "/models/model.h5")
print ('Will listen on port:', PORT, '\nAnd load model at path:', MODEL_PATH)

global model
model = keras.models.load_model(MODEL_PATH)

@app.route("/predict", methods=["GET"])
def process_image():
    file = request.files['image']
    image = np.array(img_to_mat(file.stream), ndmin=4)

    pred = model.predict(image)
    
    return 'Confidence that image matches pattern model was trained for:\n', pred[0]

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=PORT)