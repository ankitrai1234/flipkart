import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict
from scipy.spatial import distance as dist
from flask import Flask, render_template, Response

app = Flask(__name__)

# Load your YOLO model
model_path = "C:/Users/barik/OneDrive/Desktop/filpkart_grid/ml/best.pt"
model = YOLO(model_path)

# Centroid tracker dictionary
objects = defaultdict(list)
object_id = 0
max_distance = 50  # Adjust this based on your image scale and object sizes
confidence_threshold = 0.4
object_lifespan = {}
max_frames_without_detection = 15

# Capture video feed from webcam
cap = cv2.VideoCapture(0)

def generate_frames():
    global object_id
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Object detection using YOLO
        results = model(frame, conf=confidence_threshold, iou=0.2)
        centroids = []

        for result in results:
            for box in result.boxes:
                confidence = box.conf[0]
                if confidence < confidence_threshold:
                    continue
                
                x1, y1, x2, y2 = box.xyxy[0]
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                centroids.append((cx, cy))

                # Draw the bounding box and label
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = f"{model.names[int(box.cls[0])]}: {confidence:.2f}"
                cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        detected_ids = set()
        for (cx, cy) in centroids:
            matched = False
            for obj_id, centers in objects.items():
                if len(centers) > 0 and dist.euclidean((cx, cy), centers[-1]) < max_distance:
                    objects[obj_id].append((cx, cy))
                    detected_ids.add(obj_id)
                    object_lifespan[obj_id] = 0
                    matched = True
                    break

            if not matched:
                objects[object_id].append((cx, cy))
                detected_ids.add(object_id)
                object_lifespan[object_id] = 0
                object_id += 1

        # Update lifespan for each object
        for obj_id in list(objects.keys()):
            if obj_id not in detected_ids:
                object_lifespan[obj_id] += 1
                if object_lifespan[obj_id] > max_frames_without_detection:
                    del objects[obj_id]
                    del object_lifespan[obj_id]

        total_objects = len(objects)
        cv2.putText(frame, f"Total Objects: {total_objects}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode the frame to JPEG format and yield it as byte data
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
       
        # Ensure that the camera is released after the video feed ends
    cap.release()
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/index3.html')
def index1():
    return render_template('index3.html')
@app.route('/Javascriptcode.html')
def freshness():
    return render_template('Javascriptcode.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
