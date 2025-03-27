import boto3
import cv2
import os
from flask import Flask, render_template, Response, jsonify

# Initialize AWS Textract client
textract = boto3.client('textract', region_name='us-east-1')

# Initialize Flask app
app = Flask(__name__)

# Open the camera
cap = cv2.VideoCapture(0)  # Change this to 1 if you have more than one camera

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Global variable to hold extracted text
extracted_text = ""

# Function to generate frames for video feed
def generate_frames():
    global extracted_text
    while True:
        # Capture a frame from the camera
        ret, frame = cap.read()

        if ret:
            # Convert frame to bytes
            _, buffer = cv2.imencode('.png', frame)
            frame_bytes = buffer.tobytes()

            # Call Textract to detect text in the image
            response = textract.detect_document_text(Document={'Bytes': frame_bytes})

            # Extract the detected text from the response
            extracted_text = ""
            for item in response.get('Blocks', []):
                if item['BlockType'] == 'LINE':
                    extracted_text += item['Text'] + "\n"

            # Print the OCR result to the terminal
            print("Extracted Text:")
            print(extracted_text)

            # Yield frame bytes for video feed
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index2.html')  # Create a template called index.html

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/ocr_result')
def ocr_result():
    # Return the extracted text as JSON
    return jsonify(extracted_text=extracted_text)

if __name__ == '__main__':
    app.run(debug=True)

# Release the camera
cap.release()
cv2.destroyAllWindows()
