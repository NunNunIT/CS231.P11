import streamlit as st
from PIL import Image
from ultralytics import YOLO
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from custom_metric import custom_accuracy, custom_hamming_loss, custom_exact_match_ratio

# Custom metrics and LABELS remain the same
custom_objects = {
    'custom_accuracy': custom_accuracy,
    'custom_hamming_loss': custom_hamming_loss,
    'custom_exact_match_ratio': custom_exact_match_ratio
}

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
if 'camera_running' not in st.session_state:
    st.session_state.camera_running = False
if 'confidence' not in st.session_state:
    st.session_state.confidence = 0.6
if 'iou_thresh' not in st.session_state:
    st.session_state.iou_thresh = 0.2

# Load models function
@st.cache_resource(hash_funcs={tf.keras.models.Model: id})
def load_models():
    loaded_models = {}
    errors = {}
    try:
        loaded_models['cnn_model'] = load_model('./Model/cnn/cnn.h5', custom_objects=custom_objects)
    except Exception as e:
        errors['cnn_model'] = str(e)

    try:
        efficient_model = tf.keras.models.load_model('./Model/efficient/b0.h5')
        efficient_model.trainable = False
        for layer in efficient_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
        loaded_models['efficient_model'] = efficient_model
    except Exception as e:
        errors['efficient_model'] = str(e)

    try:
        loaded_models['yolo_model'] = YOLO('./Model/yolo/weight/best.pt')
    except Exception as e:
        errors['yolo_model'] = str(e)

    return loaded_models, errors

def preprocess_frame(frame, target_size=(224, 224)):
    img_resized = cv2.resize(frame, target_size)
    img_array = img_resized / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def get_predictions(model, image, threshold=0.5):
    predictions = model.predict(image)[0]
    predicted_labels = []
    for idx, prob in enumerate(predictions):
        if prob > threshold:
            predicted_labels.append((LABELS[idx], float(prob)))
    return predicted_labels

# Main layout
st.title('Fashion Object Detection')

# Sidebar
with st.sidebar:
    st.header('Input Selection')
    
    # Create tabs for image upload and camera
    tab1, tab2 = st.tabs(["Image Upload", "Camera"])
    
    with tab1:
        with st.form('file-upload-form', clear_on_submit=True):
            uploaded_img = st.file_uploader('Upload an image...', type=['jpg', 'jpeg', 'png'])
            submitted_img = st.form_submit_button('Submit/Clear Upload')

    with tab2:
        camera_option = st.selectbox('Select Camera', ['0', '1', '2', '3'])
        if st.button('Toggle Camera'):
            st.session_state.camera_running = not st.session_state.camera_running

    # Sliders for confidence and IOU
    st.session_state.confidence = st.slider('Select Model Confidence', 0, 100, 60) / 100
    st.session_state.iou_thresh = st.slider('Select IOU Threshold', 0, 100, 20) / 100

# Load models
models, load_errors = load_models()

# Main content
col1, col2 = st.columns([0.7, 0.3])

with col1:
    # Place for camera feed/results
    frame_placeholder = st.empty()

with col2:
    # Place for predictions
    predictions_placeholder = st.empty()

# Camera handling
if st.session_state.camera_running:
    try:
        cap = cv2.VideoCapture(int(camera_option))
        
        # Set properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        frame_count = 0  # Frame counter to control update frequency
        
        while cap.isOpened() and st.session_state.camera_running:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read from camera")
                break

            # Process every 5th frame to avoid excessive UI updates
            if frame_count % 5 == 0:
                # Process frame with models
                results = {}
                
                # YOLO detection
                if 'yolo_model' in models:
                    yolo_results = models['yolo_model'].predict(
                        frame,
                        conf=st.session_state.confidence,
                        iou=st.session_state.iou_thresh
                    )
                    for r in yolo_results:
                        frame = r.plot()

                # CNN predictions
                processed_frame = preprocess_frame(frame)
                if 'cnn_model' in models:
                    results['CNN'] = get_predictions(
                        models['cnn_model'], 
                        processed_frame, 
                        st.session_state.confidence
                    )

                # EfficientNet predictions
                if 'efficient_model' in models:
                    results['EfficientNet'] = get_predictions(
                        models['efficient_model'], 
                        processed_frame, 
                        st.session_state.confidence
                    )

                # Convert frame for display
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Update frame display
                frame_placeholder.image(frame, channels="RGB", use_container_width=True)
                
                # Update predictions display
                with predictions_placeholder.container():
                    st.markdown("### Predictions")
                    for model_name, preds in results.items():
                        st.markdown(f"**{model_name}:**")
                        if preds:
                            for label, prob in preds:
                                st.write(f"- {label}: {prob:.2f}")
                        else:
                            st.write("- No detections above threshold")
            
            frame_count += 1  # Increment frame counter

        # Release camera when stopped
        cap.release()
        
    except Exception as e:
        st.error(f"Error accessing camera: {str(e)}")
        st.session_state.camera_running = False