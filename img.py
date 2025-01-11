import onnxruntime as ort
import numpy as np
from PIL import Image
import streamlit as st

model_path = "C:/Users/abd/Desktop/my_model.onnx"
session = ort.InferenceSession(model_path)

st.markdown(
    """
    <style>
    .title {
        font-size: 40px;
        font-weight: bold;
        color: #4CAF50;
        text-align: center;
        padding: 20px;
    }
    .uploader {
        font-size: 20px;
        color: #333;
        text-align: center;
        margin-top: 20px;
    }
    .result {
        font-size: 30px;
        font-weight: bold;
        color: #FF5722;
        text-align: center;
        margin-top: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<p class="title">Vehicle Detection App</p>', unsafe_allow_html=True)

st.markdown('<p class="uploader">Choose an image...</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_container_width=True)
    st.write("")
    st.write("Classifying...")

    img = np.array(image.resize((180, 180))) 
    img = img / 255.0  

    if img.shape[2] == 4:
        img = img[:, :, :3]

    img = np.expand_dims(img, axis=0) 

    img = img.astype(np.float32)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    prediction = session.run([output_name], {input_name: img})

    if prediction[0][0][0] > 0.5:
        st.markdown('<p class="result">This is a <strong>Motorbike</strong></p>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="result">This is a <strong>Car</strong></p>', unsafe_allow_html=True)