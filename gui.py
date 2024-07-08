import os
import cv2
import numpy as np
from keras.models import load_model
from tkinter import Tk, Label, Button, filedialog, Canvas
from PIL import Image, ImageTk

# Load the saved model
loaded_model = load_model("facial_expression_model.h5")

# Function to load and preprocess images
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (48, 48))
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    return img

# Function to predict emotion
def predict_emotion(image_path):
    custom_test_image = preprocess_image(image_path)
    prediction = loaded_model.predict(custom_test_image)
    prediction_prob = prediction[0]
    emotion_label = np.argmax(prediction[0])
    emotion_classes = {0: 'happy', 1: 'sad', 2: 'hungry'}
    predicted_emotion = emotion_classes[emotion_label]
    return predicted_emotion, prediction_prob

# Function to open file dialog and load image
def load_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = Image.open(file_path)
        image = image.resize((200, 200))
        image = ImageTk.PhotoImage(image)
        canvas.create_image(0, 0, anchor='nw', image=image)
        canvas.image = image
        emotion, confidence = predict_emotion(file_path)
        label.config(text=f"Predicted Emotion: {emotion}\nConfidence: {confidence}")

# Create main window
root = Tk()
root.title("Animal Emotion Detector")

# Create canvas to display image
canvas = Canvas(root, width=200, height=200)
canvas.pack()

# Create label to display prediction
label = Label(root, text="", font=('Helvetica', 12))
label.pack()

# Create button to load image
button = Button(root, text="Load Image", command=load_image)
button.pack()

# Run the application
root.mainloop()
