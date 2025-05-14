from flask import Flask, request, render_template
import os
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image
from inference_sdk import InferenceHTTPClient

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize Roboflow Inference Client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="dA9ZirCb0CyiT1ng6MDq"
)
MODEL_ID = "fish-classification-jwegt-ipndw/1"  # Your model ID

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    # Handle file upload
    if 'file' in request.files and request.files['file'].filename != '':
        file = request.files['file']
        if not allowed_file(file.filename):
            return render_template('index.html', error='Invalid file type')
        
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
    
    # Handle webcam image
    elif 'webcam_image' in request.form:
        try:
            # Decode base64 image from webcam
            image_data = request.form['webcam_image'].split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes)).convert('RGB')
            filename = 'webcam_capture.jpg'
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image.save(filepath)
        except Exception as e:
            return render_template('index.html', error='Failed to process webcam image')
    
    else:
        return render_template('index.html', error='No image provided')
    
    # Perform inference with Roboflow
    try:
        result = CLIENT.infer(filepath, model_id=MODEL_ID)
        # Assuming result is a classification output with predictions
        predictions = result.get('predictions', [])
        if not predictions:
            return render_template('index.html', error='No predictions returned')
        
        # Get the top prediction
        top_prediction = max(predictions, key=lambda x: x['confidence']) if isinstance(predictions, list) else predictions
        predicted_class = top_prediction.get('class', 'Unknown')
        confidence = float(top_prediction.get('confidence', 0))
        
        return render_template('index.html', 
                             image_url=filepath,
                             prediction=predicted_class,
                             confidence=f'{confidence:.2%}')
    except Exception as e:
        return render_template('index.html', error=f'Inference failed: {str(e)}')

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)