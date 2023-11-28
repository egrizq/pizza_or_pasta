from flask import Flask, render_template, request
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img
from tensorflow.keras.preprocessing import image

import os
import numpy as np


app = Flask(__name__)
model = load_model('model.h5')


@app.route('/')
def home():
    return render_template('main.html')


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        file = request.files['file']

        if file:
            filename = file.filename
            file_path = os.path.join('static\images', filename)
            file.save(file_path)

        img = load_img(file_path, target_size=(300, 300))
        x = image.img_to_array(img)/255
        x = np.expand_dims(x, axis=0)

        images = np.vstack([x])
        classes = model.predict(images)

        if classes[0] > 0.5:
            label = "The Picture is a Pizza"
        else:
            label = "The Picture is a Pasta"

        return render_template("result.html", label=label, file_path=file_path)


if __name__ == '__main__':
    app.run(debug=True)
