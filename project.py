import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('inception.h5')

# Set up the camera
camera = cv2.VideoCapture(0)  # 0 for the default camera

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

    # Preprocess the image
    preprocessed_frame = resized_frame.astype('float32') / 255.0
    preprocessed_frame = np.expand_dims(preprocessed_frame, axis=0)

    # Perform inference using the pre-trained model
    predictions = model.predict(preprocessed_frame)
    class_index = np.argmax(predictions[0])
    label = class_labels[class_index]
    confidence = predictions[0][class_index]

    # Draw the label and confidence on the frame
    label_text = f'{label} ({confidence:.2f})'
    cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow('Leaf Disease Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
camera.release()
cv2.destroyAllWindows()