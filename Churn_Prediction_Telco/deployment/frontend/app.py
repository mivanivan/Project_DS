import json
import pandas as pd
import pickle
import streamlit as st
import requests

# load pipeline
pipe = pickle.load(open("model/preprocess_churn.pkl", "rb"))

st.title("Program Prediksi Pelanggan yang berpeluang Turnover")
gender = st.selectbox("Jenis Kelamin", ('Female', 'Male'))

senior = st.radio("Apakah pelanggan Termasuk  generasi tua ?",(0,1))
if senior == 0:
     st.write('Pelanggan < 50 tahun.')
else:
     st.write("Pelanggan > 50 tahun.")

partner = st.radio("Apakah pelanggan memiliki pasangan ?",('No','Yes'))

dependents = st.radio("Apakah pelanggan memiliki tanggungan ?",('No','Yes'))

tenor = st.number_input("Berapa lama pelanggan menggunakan jasa Telco kita ? (bulan)",min_value=1, max_value=1000, value=30, step=1)

phoneservice = st.radio("Apakah pelanggan menggunakan jasa pesawat telpon ?",('No','Yes'))

multiplelines = st.radio("Apakah pelanggan menggunakan jasa pesawat telpon multikabel untuk menerima telpon secara berasamaan ?",('No','Yes'))

internetservice = st.radio("Apakah pelanggan menggunakan jasa internet (Pilih DSL jika Bingung antara DSL & Fiber Optic ) ?",('No','DSL','Fiber optic'))

onlinesecurity = st.radio("Apakah pelanggan menggunakan jasa online security ?",('No', 'Yes', 'No internet service'))

onlinebackup = st.radio("Apakah pelanggan menggunakan jasa online backup ?",('No', 'Yes', 'No internet service'))

deviceprotection = st.radio("Apakah pelanggan menggunakan asuransi perlindungan hardware internet ?",('No', 'Yes', 'No internet service'))

techsupport = st.radio("Apakah pelanggan menggunakan jasa bantuan teknologi untuk datang ke tempatnya  ?",('No', 'Yes', 'No internet service'))

streamingtv = st.radio("Apakah pelanggan menggunakan paket streaming TV ?",('No', 'Yes', 'No internet service'))

streamingmovies = st.radio("Apakah pelanggan menggunakan paket streaming Movies ?",('No', 'Yes', 'No internet service'))

contract = st.selectbox("Perjanjian kerjasama pelanggan dengan provider ?", ['Month-to-month', 'Two year', 'One year'])

paperless = st.radio("Apakah pelanggan meminta tagihan berbentuk fisik ?",('No','Yes'))

payment = st.radio("Cara pelunasan tagihan oleh pelanggan bersangkutan ?",('Bank transfer (automatic)', 'Mailed check', 'Electronic check','Credit card (automatic)'))

bulanan = st.number_input("Biaya bulanan pelanggan ?",min_value=15, max_value=200, value=65, step=1)

totalcharges = bulanan * tenor

										
new_data = {'gender': gender,
         'SeniorCitizen': senior,
         'Partner' : partner,
         'Dependents' :dependents,
         'tenure' : tenor,
         'PhoneService' : phoneservice,
         'MultipleLines': multiplelines,
         'InternetService' : internetservice,
         'OnlineSecurity' :onlinesecurity,
         'OnlineBackup' : onlinebackup,
         'DeviceProtection' : deviceprotection,
         'TechSupport': techsupport,
         'StreamingTV' : streamingtv,
         'StreamingMovies' :streamingmovies,
         'Contract' : contract,
         'PaperlessBilling' : paperless,
         'PaymentMethod' :payment,
         'MonthlyCharges' : bulanan,
         'TotalCharges' : totalcharges
                  }
new_data = pd.DataFrame([new_data])

# build feature
new_data = pipe.transform(new_data)
new_data = new_data.tolist()

# inference
URL = "https://model-churn-backend.herokuapp.com/v1/models/churn_model:predict"
param = json.dumps({
        "signature_name":"serving_default",
        "instances":new_data
    })
r = requests.post(URL, data=param)

if r.status_code == 200:
    res = r.json()
    if res['predictions'][0][0] > 0.5:
        st.title("Pelanggan Berpotensi TURNOVER")
    else:
        st.title("Pelanggan Aman")
else:
    st.title("Unexpected Error")