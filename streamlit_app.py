import streamlit as st
from streamlit_image_select import image_select
from PIL import Image
from ultralytics import YOLO
import torch
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from custom_metric import custom_accuracy, custom_hamming_loss, custom_exact_match_ratio
import random

custom_objects = {
    'custom_accuracy': custom_accuracy,
    'custom_hamming_loss': custom_hamming_loss,
    'custom_exact_match_ratio': custom_exact_match_ratio
}

# set session state variables for persistence
if 'uploaded_img' not in st.session_state:
    st.session_state.uploaded_img = None

if 'confidence' not in st.session_state:
    st.session_state.confidence = 60 / 100

if 'iou_thresh' not in st.session_state:
    st.session_state.iou_thresh = 20 / 100

# Định nghĩa các lớp nhãn cho mô hình đa nhãn
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

# page layout
st.set_page_config(
    page_title='Fashion Object Detection',
    page_icon='👜',
    layout='wide',
    initial_sidebar_state='auto',
)

# sidebar content
with st.sidebar:
    @st.cache_data
    def load_image(image_file):
        image = Image.open(image_file)
        image = image.convert('RGB')
        return image

    st.header('Image Selection')

    with st.form('file-upload-form', clear_on_submit=True):
        st.session_state.uploaded_img = st.file_uploader('Upload an image...', type='jpg', key=1)
        submitted = st.form_submit_button('Submit/Clear Upload')

    if type(st.session_state.uploaded_img) is st.runtime.uploaded_file_manager.UploadedFile:
        st.session_state.uploaded_img = load_image(st.session_state.uploaded_img)

    st.session_state.confidence = float(st.slider(
        'Select Model Confidence', 0, 100, 60)) / 100

    st.session_state.iou_thresh = float(st.slider(
        'Select IOU Threshold', 0, 100, 20)) / 100

st.title('Fashion Object Detection')
st.caption('Select a sample fashion photo, or upload your own! - Recommended :blue[600 Height x 400 Width].')
st.caption('Afterwards, click the :blue[Detect Objects] button to see the results.')

col1, col2, col3 = st.columns([0.25, 0.25, 0.5], gap='medium')

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

with col2:
    with st.container(border=True):
        container_col2 = st.empty()

source_img = None
if st.session_state.uploaded_img is not None:
    source_img = st.session_state.uploaded_img.copy()
    st.session_state.uploaded_img = None
elif sample_img is not None:
    source_img = Image.open(sample_img)
    sample_img = None

container_col2.image(source_img,
                    caption='Input',
                    use_container_width=True
                    )

@st.cache_resource
def load_models():
    cnn_model = load_model('./Model/cnn/cnn_best.h5')
    efficient_model = load_model('./Model/cnn/cnn_best.h5')
    yolo_model = YOLO('./Model/yolo/weight/best.pt')
    return cnn_model, efficient_model, yolo_model

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
    cnn_model, efficient_model, yolo_model = load_models()
except Exception as e:
    st.error("Unable to load models. Check the specified paths.")
    st.error(e)

try:
    if st.sidebar.button('Detect Objects'):
        st.session_state.uploaded_img = None

        # Tiền xử lý ảnh một lần cho cả hai mô hình
        processed_img = preprocess_image(source_img)

        with col3:
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
            with st.container(border=True):
                # Lấy danh sách nhãn từ hàm get_predictions
                efficient_predictions = get_predictions(efficient_model, processed_img, st.session_state.confidence)
                print("eff", efficient_predictions)

                if efficient_predictions:
                    random.shuffle(efficient_predictions)  
                    for label, prob in efficient_predictions:
                        st.write(f"{label}")
                else:
                    st.write("No labels detected with confidence above threshold")

            # YOLO Model
            st.subheader('YOLO Detection')
            with st.container(border=True):
                yolo_results = yolo_model.predict(source_img,
                                                save=False,
                                                imgsz=(608, 416),
                                                conf=st.session_state.confidence,
                                                iou=st.session_state.iou_thresh
                                                )
                for r in yolo_results:
                    im_array = r.plot()
                    yolo_result = Image.fromarray(im_array[..., ::-1])
                st.image(yolo_result, caption='YOLO Result', use_container_width=True)

        source_img = None

except Exception as e:
    st.error('Error encountered during model inference.')
    st.error(e)