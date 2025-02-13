import os
import sys
sys.path.append(os.path.dirname(__file__))
import constant as const
from utils import render_box, render_mask
####
from PIL import Image
from clarifai.client import Model, Inputs
from clarifai_grpc.grpc.api import resources_pb2

import streamlit as st

# Sidebar settings
st.sidebar.header("Settings")
use_local = st.sidebar.toggle("Use local")
if use_local:
    base_url = st.sidebar.text_input("Base URL", value="http://localhost:8000")
    model_kwargs = dict(base_url=base_url, model_id="abc", user_id="1", app_id="2")
else:
    model_url = st.sidebar.text_input("Model URL", value="")
    if not model_url:
        st.error("Please insert model url")
        st.stop()
    model_kwargs = dict(url=model_url)
pat = st.sidebar.text_input("PAT", type="password")

if pat:
    os.environ["CLARIFAI_PAT"] = pat
if not os.environ.get("CLARIFAI_PAT"):
    st.error("Please insert your PAT")
    st.stop()
print(model_kwargs)
model = Model(**model_kwargs)

# Main input section
st.title("Florence any-to-any model demo")
task = st.selectbox("Select a Task", const.TASKS)
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if uploaded_image is not None:
    # Replace with API request logic
    image_bytes = uploaded_image.read()
    image_pil = Image.open(uploaded_image)
    st.image(uploaded_image, caption="Input image", width=150)
additional_prompt = st.text_area("Enter additional prompt")
max_new_tokens = st.slider("Max tokens", min_value=100, max_value=4096, value=1024)
predict_btn = st.button("Predict")


if predict_btn:
    if uploaded_image is not None:
        text_bytes = task + additional_prompt
        inputs = [Inputs.get_input_from_bytes(input_id="", text_bytes=text_bytes.encode(), image_bytes=image_bytes)]    
        # predict
        with st.spinner("Making prediction"):
            prediction = model.predict(inputs, inference_params=dict(max_new_tokens=max_new_tokens)).outputs[0]
        # Display output
        st.subheader("Output")
        with st.spinner("Rendering image"):
            if task in const.CAPTION_TASKS:
                st.text_area("", prediction.data.text.raw)
            elif task in const.DETECTION_TASKS + const.REGION_OCR_TASKS:
                image = render_box(image_pil, prediction.data.regions)
                st.image(image, width=768)
                with st.expander("Proto data "):
                    st.text_area("prediction.data.regions", prediction.data.regions)
            elif task in const.SEGMENTATION_TASKS:
                image = render_mask(image_pil, prediction.data.regions[0].region_info.mask.image.base64)
                st.image(image, width=768)
    else:    
        st.error("Please input your image")