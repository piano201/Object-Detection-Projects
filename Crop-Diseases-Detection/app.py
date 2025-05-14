from flask import Flask, render_template, request, jsonify, Response
import cv2
import os
import numpy as np
import time
from datetime import datetime
from ultralytics import YOLO
from pathlib import Path
import base64

app = Flask(__name__)

# Load the trained model
model_path = os.path.join("crop-disease-detection-using-yolov8", "Crop Disease Detection Using YOLOv8", "runs", "detect", "train3", "weights", "best.pt")
model = YOLO(model_path)

# Global variables
camera = None
confidence_threshold = 0.5
live_detections = []
detection_summary = "No detections yet"

def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
    return camera

def release_camera():
    global camera
    if camera is not None:
        camera.release()
        camera = None

def process_image(image, conf_threshold=0.5):
    """Process a single image with the YOLOv8 model and return results"""
    global live_detections, detection_summary
    
    results = model(image, conf=conf_threshold)
    
    # Get detection results
    detections = []
    disease_detections = []
    avg_confidence = 0
    
    if len(results[0].boxes) > 0:
        classes = results[0].names
        confidence_values = []
        
        for box in results[0].boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            
            if cls_id in classes:
                class_name = classes[cls_id]
                detections.append({"class": class_name, "confidence": conf})
                
                # Only count as disease if it's not just a leaf detection
                # Check if class name contains words that indicate actual diseases
                if not class_name.lower().endswith("leaf") and not class_name.lower().endswith("leafs"):
                    disease_detections.append({"class": class_name, "confidence": conf})
                    confidence_values.append(conf)
        
        if confidence_values:
            avg_confidence = sum(confidence_values) / len(confidence_values)
            
        # Update the global variables for front-end updates with actual diseases only
        live_detections = disease_detections
        num_disease_detections = len(disease_detections)
        
        if num_disease_detections > 0:
            detection_summary = f"Found {num_disease_detections} disease detection{'s' if num_disease_detections > 1 else ''}"
        else:
            detection_summary = "No diseases detected - Plant appears healthy"
    else:
        live_detections = []
        detection_summary = "No diseases detected - Plant appears healthy"
    
    # Create annotated image
    annotated_frame = results[0].plot()
    
    return {
        "annotated_image": annotated_frame,
        "detections": detections,
        "disease_detections": disease_detections,
        "num_detections": len(detections),
        "num_disease_detections": len(disease_detections),
        "avg_confidence": avg_confidence
    }

def generate_frames():
    """Generate frames for the video stream"""
    cap = get_camera()
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        # Process the frame
        result = process_image(frame, confidence_threshold)
        annotated_frame = result["annotated_image"]
        
        # Convert to JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/live_status', methods=['GET'])
def live_status():
    """Get the current detection status for the live feed"""
    return jsonify({
        "detections": live_detections,
        "summary": detection_summary
    })

@app.route('/adjust_confidence', methods=['POST'])
def adjust_confidence():
    """Adjust the confidence threshold"""
    global confidence_threshold
    data = request.json
    if 'value' in data:
        confidence_threshold = float(data['value'])
        return jsonify({"confidence_threshold": confidence_threshold})
    return jsonify({"error": "Invalid request"}), 400

@app.route('/shutdown')
def shutdown():
    """Shutdown the server and release resources"""
    release_camera()
    # Only works in development mode
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    return 'Server shutting down...'

if __name__ == '__main__':
    try:
        app.run(debug=True)
    finally:
        release_camera() 