import cv2
import numpy as np
from tensorflow.keras.models import load_model
import serial
import time

# Load the pre-trained model for leaf disease detection
leaf_model = load_model('inception.h5')

# Set up the camera
camera = cv2.VideoCapture(0)  # 0 for the default camera

# Set up serial communication
ser = serial.Serial('/dev/ttyUSB0', 9600, timeout=1)
ser.reset_input_buffer()

# Define the class labels for diseases and healthy
class_labels = [
    'Tomato___healthy',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Target_Spot',
    'Tomato___Spider_mites',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Leaf_Mold',
    'Tomato___Late_blight',
    'Tomato___Early_blight',
    'Tomato___Bacterial_spot',
    'Tomato___Tomato_mosaic_virus'
]

# Capture and process frames
while True:
    # Capture frame from the camera
    ret, frame = camera.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # Resize the frame to a smaller size for faster processing
    resized_frame = cv2.resize(frame, (224, 224))

    # Preprocess the image for leaf disease detection
    preprocessed_frame = resized_frame.astype('float32') / 255.0
    preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)

    # Perform inference for leaf disease detection using the pre-trained model
    leaf_predictions = leaf_model.predict(preprocessed_frame)
    leaf_class_index = np.argmax(leaf_predictions[0])
    leaf_label = class_labels[leaf_class_index]
    leaf_confidence = leaf_predictions[0][leaf_class_index]

    # Draw the leaf disease label and confidence on the frame
    leaf_label_text = f'{leaf_label} ({leaf_confidence:.2f})'
    cv2.putText(frame, leaf_label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Leaf Disease Detection', frame)

    # Control device based on key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        print('Sending command for action 1...')
        ser.write(b"1\n")
        time.sleep(1)
    elif key == ord('d'):
        print('Sending command for action 4...')
        ser.write(b"4\n")
        time.sleep(1)

# Release resources
camera.release()
cv2.destroyAllWindows()
