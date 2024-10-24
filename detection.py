import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Load the model
model = load_model('inception.h5')

# Open the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Preprocess function to resize the image
def preprocess_image(img):
    resized_img = cv2.resize(img, (224, 224))  # Resize the image to match the model's input shape
    resized_img = np.expand_dims(resized_img, axis=0)  # Add batch dimension
    return resized_img

while True:
    # Read a frame from the webcam
    success, img = cap.read()
    
    # Preprocess the image
    preprocessed_img = preprocess_image(img)
    
    # Perform inference
    results = model.predict(preprocessed_img)
    
    # Process the results
    # (Your code for processing results goes here)
    
    # Display the result
    cv2.imshow('Result', img)
    
    # Quit on 'q' key press
    if cv2.waitKey(1) == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

