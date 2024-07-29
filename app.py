from flask import Flask, render_template, request
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import imutils    
app = Flask(__name__)

# Load the pre-trained model
model = keras.models.load_model('model.h5')

# List of class names for the model output
class_names = ['glioma_tumor', 'meningioma_tumor', 'no_tumor', 'pituitary_tumor']

@app.route('/info', methods=['GET', 'POST'])
def info():
    return render_template('info.html')

@app.route('/', methods=['GET', 'POST'])
def upload_and_predict():
    if request.method == 'POST':
        # Get the uploaded image file
        image_file = request.files['image']

        if image_file:
            # Read and preprocess the image using OpenCV
            image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_COLOR)
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
    
            img_thresh = cv2.threshold(img_gray, 45, 255, cv2.THRESH_BINARY)[1]
            img_thresh = cv2.erode(img_thresh, None, iterations=2)
            img_thresh = cv2.dilate(img_thresh, None, iterations=2)

            contours = cv2.findContours(img_thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = imutils.grab_contours(contours)
            c = max(contours, key=cv2.contourArea)

            extLeft = tuple(c[c[:, :, 0].argmin()][0])
            extRight = tuple(c[c[:, :, 0].argmax()][0])
            extTop = tuple(c[c[:, :, 1].argmin()][0])
            extBot = tuple(c[c[:, :, 1].argmax()][0])
    
            image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            image = cv2.resize(image, (224, 224))  # Resize the image
            # image = image / 255.0  # Normalize the image

            # Add batch dimension
            image = np.expand_dims(image, axis=0)

            # Perform classification
            prediction = model.predict(image)
            class_index = np.argmax(prediction)
            class_label = class_names[class_index]

            return render_template('result.html', class_label=class_label, )

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
