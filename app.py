import streamlit as st
import PIL
from PIL import Image
import numpy as np
from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import zipfile
import io
import time

tab1, tab2 = st.tabs(["Face Detection", "Images"])

with tab1:
    col1, col2 = st.columns(2, gap='small')
    with col2:
        mainimg = Image.open('logo.png')
        st.image(mainimg, width=90)
    with col1:
        st.title('Recognify')
    st.logo('logo.png')

    st.title(" ")

    up_col, ref_col = st.columns(2, gap='large')
    with up_col:
        upload = st.file_uploader("Files to compare", accept_multiple_files=True)
    with ref_col:
        reference = st.file_uploader("Upload Your Face")

    def status():
        st.markdown("""
        <style>
        .stProgress .st-bo {
        background-color: Red;
        }
        </style>""", unsafe_allow_html=True)

        progress_text = "Operation in progress. Please wait."
        my_bar = st.progress(0, text=progress_text)

        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1, text=progress_text)
        time.sleep(1)
        my_bar.empty()

    def create_zip(images, filenames):
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'a', zipfile.ZIP_DEFLATED, False) as zip_file:
            for img, filename in zip(images, filenames):
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='PNG')
                img_buffer.seek(0)
                zip_file.writestr(filename, img_buffer.read())
        zip_buffer.seek(0)
        return zip_buffer

    verified_img = []
    verified_File_name = []
    compare_btn = st.button("Detect")

    if compare_btn:
        try:
            reference_img = Image.open(reference)
            img1 = np.array(reference_img)
            for images in upload:
                try:
                    img2 = np.array(Image.open(images))
                    verify = DeepFace.verify(img1, img2)
                    status()
                    if verify['verified']:
                        st.success(f"Face Identified in \"{images.name}\" ")
                        verified_img.append(Image.open(images))
                        verified_File_name.append(images.name)
                    else:
                        st.error(f"Face not Identified in \"{images.name}\" ")
                except Exception as e:
                    st.error(f"Error processing image {images.name}: {e}")
        except Exception as e:
            st.error(f"Error processing reference image: {e}")

    zip_file = create_zip(verified_img, verified_File_name)

with tab2:
    st.image(verified_img, width=150)
    st.download_button(
        "Download recognized images",
        data=zip_file,
        file_name='Recognized.zip',
        mime="application/zip"
    )
