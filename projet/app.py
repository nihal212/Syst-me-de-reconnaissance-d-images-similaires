from flask import Flask, render_template, request, jsonify,send_from_directory
import cv2
import numpy as np
from numpy.linalg import norm
import pickle
from sklearn.neighbors import NearestNeighbors
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.layers import GlobalMaxPooling2D
import tensorflow as tf
from werkzeug.utils import secure_filename
import os

# Load precomputed feature vectors and filenames
feature_list = np.array(pickle.load(open("featurevector2.pkl", "rb")))
filename = np.array(pickle.load(open("filenames2.pkl", "rb")))

# Load ResNet50 model
model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model.trainable = False

model = tf.keras.Sequential([
    model,
    GlobalMaxPooling2D()
])
model.summary()

# Create Nearest Neighbors model
neighbors = NearestNeighbors(n_neighbors=8, algorithm="brute", metric="euclidean")
neighbors.fit(feature_list)

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')
@app.route('/dataset/<filename>')
def get_image(filename):
    return send_from_directory('dataset', filename)

@app.route('/process_image', methods=['POST'])
def process_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})

    file = request.files['file']

    # Check if the file has a valid extension
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif'}
    file_extension = os.path.splitext(file.filename)[1].lower()
    if file_extension not in allowed_extensions:
        return jsonify({'error': 'Invalid file extension. Allowed extensions: jpg, jpeg, png, gif'})

    # Save the file with a secure filename
    filenam = secure_filename(file.filename)
    file_path = os.path.join("static", filenam)
    file.save(file_path)

    # Use cv2.imread() to read the image
    img = cv2.imread(file_path)

    if img is None or img.size == 0:
        return jsonify({'error': f"Error: Unable to read the image from the path {file_path} or empty image."})

    # Add a check to ensure the image is not empty before resizing
    if img.size == 0:
        return jsonify({'error': 'Error: Empty image.'})

    img = cv2.resize(img, (224, 224))
    img = np.array(img)
    expand_img = np.expand_dims(img, axis=0)
    pre_img = preprocess_input(expand_img)
    result = model.predict(pre_img).flatten()
    normalized = result / norm(result)

    distances, indices = neighbors.kneighbors([normalized])
    # Debugging lines
    similar_images = [filename[file] for file in indices[0][0:10]]


    # Get filenames of similar images
    #similar_images = [filename[file] for file in indices[0][1:8]]

    # Debugging lines

    # Now, similar_images contains the modified paths

    # Return the list of similar images
    return jsonify({'similar_images': similar_images})



if __name__ == '__main__':
    app.run(debug=True)




