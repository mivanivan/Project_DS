import json
import numpy as np
import streamlit as st
import requests
from keras.applications import VGG16
from PIL import Image


st.title("Program Deteksi Gender Berdasarkan Foto Wajah")
uploaded_files_0 = st.file_uploader("Choose your close-up photos", type=['jpg','png','jpeg'])

if uploaded_files_0 is not None:
    image          = Image.open(uploaded_files_0)
    st.image(image)

if st.button ('Predict'):
    uploaded_files = Image.open(uploaded_files_0).resize((96,96))   
    uploaded_files = np.array(uploaded_files)    
    data           = uploaded_files[np.newaxis,...]
    model          = VGG16(include_top=False, weights='imagenet',input_shape=(96,96,3))
    new_data       = model.predict(data)
    new_data       = new_data.tolist()
     
    # inference
    URL = "https://model-gender-backend.herokuapp.com/v1/models/gender_model:predict"
    param = json.dumps({
            "signature_name":"serving_default",
            "instances":new_data
        })
    r = requests.post(URL, data=param)

    if r.status_code == 200:
        res = r.json()
        if res['predictions'][0][0] > 0.5:
            st.title("Foto Pria")
        else:
            st.title("Foto Wanita")
    else:
        st.title("Unexpected Error")
else:
    st.write('Lakukan Upload Gambar terlebih Dahulu')


