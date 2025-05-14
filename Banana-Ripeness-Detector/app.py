from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from PIL import Image
import os
import shutil

app = Flask(__name__)

# Upload folder configuration
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load YOLOv8 model
model = YOLO('C:/Users/josep/OneDrive/Desktop/microfinals/best.pt')  # Adjust path if needed

@app.route('/', methods=['GET', 'POST'])
def index():
    result_img = None
    classification = None
    confidence = None

    if request.method == 'POST':
        if 'image' not in request.files:
            return render_template('index.html', result_img=None)

        file = request.files['image']
        if file.filename == '':
            return render_template('index.html', result_img=None)

        if file:
            # Secure and convert filename
            original_filename = secure_filename(file.filename)
            name_only = os.path.splitext(original_filename)[0]
            jpg_filename = f"{name_only}.jpg"
            jpg_path = os.path.join(app.config['UPLOAD_FOLDER'], jpg_filename)

            # Convert and save image as JPG
            img = Image.open(file.stream).convert('RGB')
            img.save(jpg_path, 'JPEG')

            # Run YOLOv8 prediction
            results = model.predict(source=jpg_path, save=True, save_txt=False, conf=0.3)
            result_path = os.path.join(results[0].save_dir, jpg_filename)

            # Move result image to uploads
            result_filename = f"result_{jpg_filename}"
            result_image_path = os.path.join(UPLOAD_FOLDER, result_filename)
            shutil.move(result_path, result_image_path)

            # Get classification result
            labels = results[0].names
            boxes = results[0].boxes
            classes = boxes.cls.tolist()
            confidences = boxes.conf.tolist()

            if classes:
                detected_class = labels[int(classes[0])]
                detected_confidence = confidences[0] * 100
                print(f"Detected: {detected_class} ({detected_confidence:.2f}%)")
                classification = detected_class
                confidence = detected_confidence

            return render_template(
                'index.html',
                result_img=result_filename,
                classification=classification,
                confidence=confidence
            )

    return render_template('index.html', result_img=None)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
