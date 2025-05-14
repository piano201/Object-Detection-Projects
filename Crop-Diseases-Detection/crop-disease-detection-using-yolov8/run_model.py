from ultralytics import YOLO
import cv2
import os
import sys
import argparse
import numpy as np
import time
from datetime import datetime

# Load the trained model
model_path = r"d:\Code\google colab\crop-disease-detection-using-yolov8\Crop Disease Detection Using YOLOv8\runs\detect\train3\weights\best.pt"
model = YOLO(model_path)

# Create output directories if they don't exist
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "crop_detection_results")
screenshot_dir = os.path.join(output_dir, "screenshots")
video_dir = os.path.join(output_dir, "videos")

for directory in [output_dir, screenshot_dir, video_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def process_webcam(output_path=None, auto_record=True):
    """Process webcam feed for real-time crop disease detection"""
    # Open the webcam (0 is usually the default camera)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create automatic output path if not provided
    if auto_record and not output_path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(video_dir, f"crop_detection_{timestamp}.mp4")
        print(f"Auto-recording video to: {output_path}")
    
    # Setup video writer if output path is provided
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        recording = True
    else:
        recording = False
    
    print("Webcam is active for crop disease detection.")
    print("Press 'q' to quit, 's' to save a screenshot, 'r' to toggle recording.")
    
    # Create a named window and set it to full screen
    window_name = "Crop Disease Detection - Real-time Analysis"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Colors for UI elements
    header_color = (40, 40, 45)
    text_color = (240, 240, 240)
    healthy_color = (0, 200, 0)
    warning_color = (0, 165, 255)  # Orange in BGR
    recording_color = (0, 0, 255)  # Red in BGR
    
    # Initialize variables
    start_time = time.time()
    frame_count = 0
    fps_display = 0
    screenshot_saved = False
    screenshot_msg_time = 0
    confidence_threshold = 0.5  # Default confidence threshold
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # FPS calculation
        frame_count += 1
        elapsed_time = time.time() - start_time
        if elapsed_time >= 1.0:
            fps_display = frame_count / elapsed_time
            frame_count = 0
            start_time = time.time()
            
        # Create a copy of the frame for display
        display_frame = frame.copy()
        
        # Run inference
        results = model(frame, conf=confidence_threshold)
        
        # Visualize results
        annotated_frame = results[0].plot()
        
        # Add stylish header with instructions
        header_height = 70
        header = np.zeros((header_height, annotated_frame.shape[1], 3), dtype=np.uint8)
        header[:] = header_color
        
        # Add horizontal line at bottom of header
        header[header_height-2:header_height, :] = (100, 100, 100)
        
        # Add text to header
        cv2.putText(header, "Crop Disease Detection System", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, text_color, 2)
                   
        cv2.putText(header, f"FPS: {fps_display:.1f}", (20, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
        
        # Show recording status in header
        if recording:
            # Add recording indicator (flashing red dot)
            if int(time.time() * 2) % 2 == 0:  # Flashing effect
                cv2.circle(header, (annotated_frame.shape[1] - 20, 30), 8, recording_color, -1)
            cv2.putText(header, "REC", (annotated_frame.shape[1] - 50, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, recording_color, 2)
        
        # Show controls
        controls_text = "q:quit | s:screenshot | r:toggle rec"
        cv2.putText(header, controls_text, 
                   (annotated_frame.shape[1] - 280, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
                   
        cv2.putText(header, f"Conf: {confidence_threshold:.2f} (+/- to adjust)", 
                   (annotated_frame.shape[1] - 280, 55), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
        
        # Add footer with detection info
        footer_height = 100
        footer = np.zeros((footer_height, annotated_frame.shape[1], 3), dtype=np.uint8)
        footer[:] = header_color
        
        # Add horizontal line at top of footer
        footer[0:2, :] = (100, 100, 100)
        
        # Get detection results
        if len(results[0].boxes) > 0:
            num_detections = len(results[0].boxes)
            classes = results[0].names
            detected_classes = []
            confidence_values = []
            
            # Collect unique class names and confidence values
            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                
                if cls_id in classes:
                    class_name = classes[cls_id]
                    if class_name not in detected_classes:
                        detected_classes.append(class_name)
                    confidence_values.append(conf)
            
            # Calculate average confidence if detections exist
            avg_confidence = sum(confidence_values) / len(confidence_values) if confidence_values else 0
            
            # Display detection information
            status_text = f"STATUS: {num_detections} disease detection(s)"
            cv2.putText(footer, status_text, (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, warning_color, 2)
            
            # Display detected classes with confidence
            class_text = "Detected: " + ", ".join(detected_classes)
            cv2.putText(footer, class_text, (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
                       
            cv2.putText(footer, f"Avg. Confidence: {avg_confidence:.2f}", (20, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 1)
            
            # Show save path info on the right side
            if recording:
                save_dir_text = f"Saving to: {os.path.basename(output_path)}"
                cv2.putText(footer, save_dir_text, 
                          (annotated_frame.shape[1] - 350, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
        else:
            cv2.putText(footer, "STATUS: No diseases detected", (20, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, healthy_color, 2)
            cv2.putText(footer, "Plant appears healthy", (20, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            cv2.putText(footer, "Keep monitoring for any changes", (20, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 1)
            
            # Show save path info on the right side
            if recording:
                save_dir_text = f"Saving to: {os.path.basename(output_path)}"
                cv2.putText(footer, save_dir_text, 
                          (annotated_frame.shape[1] - 350, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)
        
        # Add screenshot saved message if needed
        if screenshot_saved and time.time() - screenshot_msg_time < 3:  # Show for 3 seconds
            cv2.putText(annotated_frame, "Screenshot Saved!", 
                      (annotated_frame.shape[1]//2 - 100, annotated_frame.shape[0]//2), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        
        # Combine header, annotated frame, and footer
        combined_frame = np.vstack((header, annotated_frame, footer))
        
        # Write to output video if recording
        if recording:
            # Resize to original dimensions for video saving
            save_frame = cv2.resize(annotated_frame, (width, height))
            out.write(save_frame)
        
        # Display the combined frame
        cv2.imshow(window_name, combined_frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save screenshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            screenshot_filename = os.path.join(screenshot_dir, f"crop_detection_{timestamp}.jpg")
            cv2.imwrite(screenshot_filename, combined_frame)
            screenshot_saved = True
            screenshot_msg_time = time.time()
            print(f"Screenshot saved as {screenshot_filename}")
        elif key == ord('r'):
            # Toggle recording
            if recording:
                out.release()
                recording = False
                print(f"Stopped recording video: {output_path}")
            else:
                # Create new output file
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = os.path.join(video_dir, f"crop_detection_{timestamp}.mp4")
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                recording = True
                print(f"Started recording video: {output_path}")
        elif key == ord('+') or key == ord('='):
            # Increase confidence threshold (max 0.95)
            confidence_threshold = min(0.95, confidence_threshold + 0.05)
        elif key == ord('-') or key == ord('_'):
            # Decrease confidence threshold (min 0.05)
            confidence_threshold = max(0.05, confidence_threshold - 0.05)
    
    # Release resources
    cap.release()
    if recording:
        out.release()
    cv2.destroyAllWindows()
    
    print(f"\nVideos saved to: {video_dir}")
    print(f"Screenshots saved to: {screenshot_dir}")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Crop Disease Detection using YOLOv8')
    parser.add_argument('--save', action='store_true', help='Save output video')
    parser.add_argument('--output', type=str, help='Output path for the processed video')
    parser.add_argument('--no-auto-record', action='store_true', help='Disable automatic video recording')
    return parser.parse_args()

if __name__ == "__main__":
    print("Starting Crop Disease Detection System")
    print(f"Results will be saved to: {output_dir}")
    print("Opening webcam for real-time analysis...")
    
    # Check if command line arguments are provided
    if len(sys.argv) > 1:
        args = parse_arguments()
        auto_record = not args.no_auto_record
        
        if args.save and args.output:
            process_webcam(args.output, auto_record)
        else:
            process_webcam(None, auto_record)
    else:
        # No command arguments, just start the webcam with auto-recording
        process_webcam()
