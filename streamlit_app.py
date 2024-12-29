import streamlit as st
from streamlit_image_select import image_select
from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import tempfile
import os
from custom_metric import custom_accuracy, custom_hamming_loss, custom_exact_match_ratio
import random
import joblib
from scipy.sparse import issparse
from visualizeFunc import predict_image, predict_image_from_ml

# Custom metrics
custom_objects = {
    'custom_accuracy': custom_accuracy,
    'custom_hamming_loss': custom_hamming_loss,
    'custom_exact_match_ratio': custom_exact_match_ratio
}

# Labels
LABELS = [
    "shirt, blouse",
    "top, t-shirt, sweatshirt",
    "jacket",
    "pants",
    "skirt",
    "dress",
    "shoe",
    "bag, wallet"
]

# Streamlit setup
st.set_page_config(
    page_title='Fashion Object Detection',
    page_icon='👜',
    layout='wide',
    initial_sidebar_state='auto',
)

# Session state variables
if 'uploaded_img' not in st.session_state:
    st.session_state.uploaded_img = None
if 'confidence' not in st.session_state:
    st.session_state.confidence = 0.6
if 'iou_thresh' not in st.session_state:
    st.session_state.iou_thresh = 0.2

# Sidebar
with st.sidebar:
    @st.cache_data
    def load_image(image_file):
        image = Image.open(image_file)
        image = image.convert('RGB')
        return image

    st.header('Image Selection')
    with st.form('file-upload-form', clear_on_submit=True):
        st.session_state.uploaded_img = st.file_uploader('Upload an image...', type=['jpg', 'jpeg', 'png'])
        submitted = st.form_submit_button('Submit/Clear Upload')

    if isinstance(st.session_state.uploaded_img, st.runtime.uploaded_file_manager.UploadedFile):
        st.session_state.uploaded_img = load_image(st.session_state.uploaded_img)

    st.session_state.confidence = st.slider(
        'Select Model Confidence', 0, 100, 60) / 100
    st.session_state.iou_thresh = st.slider(
        'Select IOU Threshold', 0, 100, 20) / 100

# Main layout
st.title('Fashion Object Detection')
st.caption('Select a sample fashion photo, or upload your own!')
st.caption('Click the :blue[Detect Objects] button to see the results.')

col1, col2, col3 = st.columns([0.25, 0.25, 0.5], gap='medium')

# Sample image selection
with col1:
    sample_img = image_select(
        label='Select a sample fashion photo',
        images=[
            './sample_images/1.jpg',
            './sample_images/2.jpg',
            './sample_images/3.jpg',
            './sample_images/4.jpg',
            './sample_images/5.jpg',
            './sample_images/6.jpg',
        ],
        captions=['Sample #1', 'Sample #2', 'Sample #3', 'Sample #4', 'Sample #5', 'Sample #6'],
        use_container_width=False
    )

# Image display container
with col2:
    container_col2 = st.empty()

source_img = None
if st.session_state.uploaded_img is not None:
    source_img = st.session_state.uploaded_img
    st.session_state.uploaded_img = None
elif sample_img is not None:
    source_img = Image.open(sample_img)

if source_img:
    container_col2.image(source_img, caption='Input', use_container_width=True)

# Load models
@st.cache_resource(hash_funcs={tf.keras.models.Model: id})
def load_models():
    cnn_model = load_model('./Model/cnn/cnn.h5', custom_objects=custom_objects)
    # Load EfficientNet model with proper configuration
    efficient_model = tf.keras.models.load_model('./Model/efficient/b0.h5')
    # Set to inference mode
    efficient_model.trainable = False
    for layer in efficient_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
            
    # Compile the model to ensure proper setup
    efficient_model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=[custom_accuracy, custom_hamming_loss, custom_exact_match_ratio]
    ) 
    yolo_model = YOLO('./Model/yolo/weight/best.pt')
    best_ml = joblib.load('./Model/ml/model__64__hog__17.joblib')
    return cnn_model, efficient_model, yolo_model, best_ml

def preprocess_image(image, target_size=(224, 224)):
    # Match exactly your notebook preprocessing
    if isinstance(image, Image.Image):
        # Convert PIL Image to cv2 format
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    img_resized = cv2.resize(image, target_size)
    img_array = img_resized / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
    

def get_predictions(model, image, threshold=0.5):
    predictions = model.predict(image)[0]  # Get first element like in notebook
    # Use the same logic as your notebook
    predicted_labels = []
    for idx, prob in enumerate(predictions):
        if prob > threshold:  # Use same threshold comparison
            predicted_labels.append((LABELS[idx], float(prob)))
    
    # Add debug prints
    print("Raw predictions:", predictions)
    print("Thresholded indices:", [i for i, prob in enumerate(predictions) if prob > threshold])
    return predicted_labels


try:
    cnn_model, efficient_model, yolo_model, ml_model = load_models()
except Exception as e:
    st.error("Unable to load models. Check the specified paths.")
    st.error(e)

# Object detection
if st.sidebar.button('Detect Objects') and source_img is not None:
    processed_img = preprocess_image(source_img)
    try:
        with col3:
            # ML Model
            st.subheader('ML-KMM Model Predictions')
            with st.container(border=True):
                image = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
                ml_predictions = predict_image_from_ml(ml_model, image, st.session_state.confidence)
                print("ML-KNN", ml_predictions)
                if len(ml_predictions) > 0:
                    for label, prob in ml_predictions:
                        st.write(f"{label}")
                else:
                    st.write("No labels detected with confidence above threshold")

            # CNN Model
            st.subheader('CNN Model Predictions')
            with st.container(border=True):
                cnn_predictions = get_predictions(cnn_model, processed_img, st.session_state.confidence)
                print("CNN", cnn_predictions)
                if cnn_predictions:
                    if len(cnn_predictions) > 1:
                        cnn_predictions = cnn_predictions[:-1]

                    for label, prob in cnn_predictions:
                        st.write(f"{label}")
                else:
                    st.write("No labels detected with confidence above threshold")

            # EfficientNet Model
            st.subheader('EfficientNet Predictions')
            efficient_predictions = predict_image(source_img, efficient_model, st.session_state.confidence)
            print("EFF", efficient_predictions)
            if efficient_predictions:
                for label, prob in efficient_predictions:
                    st.write(f"{label}: {prob:.2f}")
            else:
                st.write("No labels detected with confidence above threshold")

            # YOLO Model
            st.subheader('YOLO Detection')
            # Create temporary file for YOLO processing if needed
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                img_path = tmp_file.name
                source_img.save(img_path)
                
            yolo_results = yolo_model.predict(
                img_path,
                save=False,
                imgsz=(608, 416),
                conf=st.session_state.confidence,
                iou=st.session_state.iou_thresh
            )
            
            for r in yolo_results:
                im_array = r.plot()
                yolo_result = Image.fromarray(im_array[..., ::-1])
            st.image(yolo_result, caption='YOLO Result', use_container_width=True)
            
            # Clean up temporary file
            os.unlink(img_path)

    except Exception as e:
        st.error('Error encountered during model inference.')
        st.error(e)
else:
    if st.sidebar.button('Detect Objects'):
        st.error('Please select or upload an image first.')