from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.preprocessing import image

import os
import numpy as np

app = Flask(__name__)
model = load_model('model_pizza.h5')
target_img = os.path.join(os.getcwd(), 'static\images')

# TODO HOMEPAGE


@app.route('/home')
def home():
    return render_template('index.html')


# TODO FORMAT CHECK
FORMAT = set(['jpg', 'png', 'jpeg'])


def allowed_filename(file):
    reformat = file.split('.')[1]
    if reformat in FORMAT:
        return True

# TODO PREDICT


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']  # todo contains images from html
        # todo read the formated file is true and 'file' != none
        if file and allowed_filename(file.filename):
            filename = file.filename
            file_path = os.path.join('static\images', filename)
            file.save(file_path)  # todo saving the images

            img = load_img(file_path, target_size=(300, 300))
            x = image.img_to_array(img) / 255
            x = np.expand_dims(x, axis=0)

            images = np.vstack([x])
            # todo model predict the image
            classes = model.predict(images, batch_size=10)

            if classes[0] > 0.5:
                label = "The picture is a pizza"
            else:
                label = "The picture is a pasta"

            return render_template('predict.html', label=label, user_image=file_path)

        else:
            return render_template('wrong.html')


if __name__ == '__main__':
    app.run(debug=True)
