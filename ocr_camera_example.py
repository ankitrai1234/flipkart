import boto3
import cv2
import os
from PIL import Image

# Initialize AWS Textract client
textract = boto3.client('textract')

# Open the camera
cap = cv2.VideoCapture(0)  # Change this to 1 if you have more than one camera


if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

print("Camera opened successfully. Press 'q' to quit.")

# Create/open a file to save OCR results
output_file = 'ocr_output_aws.txt'
with open(output_file, 'w') as f:
    f.write("AWS Textract OCR Results:\n")

while True:
    # Capture a frame from the camera
    ret, frame = cap.read()

    if ret:
        # Save the captured frame as an image
        image_path = 'captured_image.png'
        cv2.imwrite(image_path, frame)

        # Open the image file
        with open(image_path, 'rb') as document:
            # Call Textract to detect text in the image
            response = textract.detect_document_text(Document={'Bytes': document.read()})

        # Extract the detected text from the response
        extracted_text = ""
        for item in response['Blocks']:
            if item['BlockType'] == 'LINE':
                extracted_text += item['Text'] + "\n"

        # Print the OCR result to the terminal
        print("Extracted Text:")
        print(extracted_text)

        # Save the OCR result to the file
        with open(output_file, 'a') as f:
            f.write(extracted_text + "\n")

        # Show the frame in a window (optional)
        cv2.imshow('Live OCR', frame)

    # Press 'q' to quit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()

# Optionally, delete the temporary image file
if os.path.exists(image_path):
    os.remove(image_path)
