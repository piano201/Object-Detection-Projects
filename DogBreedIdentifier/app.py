from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pickle
import os

app = Flask(__name__)

model = load_model('model/dogBreedModel2.h5')

with open('breed_labels.pkl', 'rb') as f:
    breed_labels = pickle.load(f)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    filename = None

    if request.method == 'POST':
        file = request.files['image']  
        if file:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            filename = file.filename

            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            preds = model.predict(img_array)
            top_idx = np.argmax(preds[0])
            prediction = breed_labels[top_idx]
            confidence = round(preds[0][top_idx] * 100, 2)

    return render_template('index.html', prediction=prediction, confidence=confidence, filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
