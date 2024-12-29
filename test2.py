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
import joblib
from skimage.feature import hog
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
    page_icon='💼',
    layout='wide',
    initial_sidebar_state='expanded',
)

# Session state variables
if 'uploaded_img' not in st.session_state:
    st.session_state.uploaded_img = None
if 'camera_running' not in st.session_state:
    st.session_state.camera_running = False
if 'confidence' not in st.session_state:
    st.session_state.confidence = 0.6
if 'iou_thresh' not in st.session_state:
    st.session_state.iou_thresh = 0.2

# Load models
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

    try:
        loaded_models['ml_model'] = joblib.load('./Model/ml/model__64__hog__17.joblib')
    except Exception as e:
        errors['ml_model'] = str(e)

    return loaded_models, errors

models, load_errors = load_models()

# Sidebar
with st.sidebar:
    tabs = st.tabs(["Upload Image", "Video Capture"])

    # Tab 1: Upload Image
    with tabs[0]:
        # st.session_state.uploaded_img = st.file_uploader('Upload an image...', type=['jpg', 'jpeg', 'png'])
        # if st.session_state.uploaded_img:
        #     st.session_state.uploaded_img = Image.open(st.session_state.uploaded_img)
        # # st.slider('Confidence Threshold', 0, 100, 60, key='confidence_slider')
        # st.session_state.confidence = st.slider(
        # 'Select Model Confidence', 0, 100, 60) / 100
        # st.session_state.iou_thresh = st.slider(
        #     'Select IOU Threshold', 0, 100, 20) / 100
        # st.session_state.camera_running = False
        @st.cache_data
        def load_image(image_file):
            image = Image.open(image_file)
            image = image.convert('RGB')
            return image

        with st.form('file-upload-form', clear_on_submit=True):
            st.session_state.uploaded_img = st.file_uploader('Upload an image...', type=['jpg', 'jpeg', 'png'])
            submitted = st.form_submit_button('Submit/Clear Upload')

        if isinstance(st.session_state.uploaded_img, st.runtime.uploaded_file_manager.UploadedFile):
            st.session_state.uploaded_img = load_image(st.session_state.uploaded_img)

    # Tab 2: Video Capture
    with tabs[1]:
        camera_option = st.selectbox('Select Camera', ['0', '1', '2', '3'])
        if st.button('Toggle Camera'):
            st.session_state.camera_running = True
    
    st.session_state.confidence = st.slider(
    'Select Model Confidence', 0, 100, 60) / 100
    st.session_state.iou_thresh = st.slider(
    'Select IOU Threshold', 0, 100, 20) / 100

# Main layout
st.title('Fashion Object Detection')

# Tab 1: Upload Image
if not st.session_state.camera_running:
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

        if load_errors:
            with col3:
                for model_name, error_message in load_errors.items():
                    st.error(f"{model_name} could not be loaded: {error_message}")

        if st.button('Detect Objects'):
            with col3:
                if 'cnn_model' in models:
                    try:
                        st.subheader('CNN Model Predictions')
                        cnn_predictions = predict_image(source_img, models['cnn_model'], st.session_state.confidence)
                        if cnn_predictions:
                            for label, prob in cnn_predictions:
                                st.markdown(f"<p style='font-size:20px;'>{label}: {prob}</p>", unsafe_allow_html=True)
                        else:
                            st.write("No labels detected with confidence above threshold")
                    except Exception as e:
                        st.error(f"Error in CNN model: {e}")

                if 'efficient_model' in models:
                    try:
                        st.subheader('EfficientNet Predictions')
                        efficient_predictions = predict_image(source_img, models['efficient_model'], st.session_state.confidence)
                        if efficient_predictions:
                            for label, prob in efficient_predictions:
                                st.markdown(f"<p style='font-size:20px;'>{label}: {prob}</p>", unsafe_allow_html=True)
                        else:
                            st.write("No labels detected with confidence above threshold")
                    except Exception as e:
                        st.error(f"Error in EfficientNet model: {e}")

                if 'yolo_model' in models:
                    try:
                        st.subheader('YOLO Detection')
                        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
                            img_path = tmp_file.name
                            source_img.save(img_path)
                        yolo_results = models['yolo_model'].predict(
                            img_path,
                            save=False,
                            imgsz=(608, 416),
                            conf=st.session_state.confidence
                        )
                        for r in yolo_results:
                            im_array = r.plot()
                            yolo_result = Image.fromarray(im_array[..., ::-1])
                        st.image(yolo_result, caption='YOLO Result', use_container_width=True)
                        os.unlink(img_path)
                    except Exception as e:
                        st.error(f"Error in YOLO model: {e}")
# Tab 2: Video Capture
else:
    col1, col2 = st.columns([0.75, 0.25], gap='medium')

    with col1:
        frame_placeholder = st.empty()

    with col2:
        predictions_placeholder = st.empty()

    if st.session_state.camera_running:
        try:
            cap = cv2.VideoCapture(int(camera_option))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)

            frame_count = 0

            while cap.isOpened() and st.session_state.camera_running:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read from camera")
                    break

                if frame_count % 5 == 0:
                    results = {}

                    if 'yolo_model' in models:
                        yolo_results = models['yolo_model'].predict(
                            frame,
                            conf=st.session_state.confidence
                        )
                        for r in yolo_results:
                            frame = r.plot()

                    if 'cnn_model' in models:
                        results['CNN'] = predict_image(
                            frame, 
                            models['cnn_model'], 
                            st.session_state.confidence
                        )

                    if 'efficient_model' in models:
                        results['EfficientNet'] = predict_image(
                            frame, 
                            models['efficient_model'], 
                            st.session_state.confidence
                        )

                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(frame, channels="RGB", use_container_width=True)

                    with predictions_placeholder.container():
                        st.markdown("### Predictions")
                        for model_name, preds in results.items():
                            st.markdown(f"**{model_name}:**")
                            if preds:
                                for label, prob in preds:
                                    st.write(f"- {label}: {prob:.2f}")
                            else:
                                st.write("- No detections above threshold")

                frame_count += 1

            cap.release()
        except Exception as e:
            st.error(f"Error accessing camera: {str(e)}")
            st.session_state.camera_running = False
