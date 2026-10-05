import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("music_genre_model.pkl")

st.set_page_config(page_title="Music Genre Classifier", page_icon="🎵")

st.title("Music Genre Classifier")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    try:
        predictions = model.predict(data)
        data["Predicted Genre"] = predictions

        st.subheader("Predictions")
        st.dataframe(data, use_container_width=True)

        st.subheader("Genre Summary")
        st.bar_chart(data["Predicted Genre"].value_counts())
    except Exception as e:
        st.error(str(e))

st.subheader("Single Song Prediction")

feature_names = list(model.feature_names_in_)
values = []

for feature in feature_names:
    values.append(st.number_input(feature, value=0.0))

if st.button("Predict Genre"):
    input_data = pd.DataFrame([values], columns=feature_names)
    prediction = model.predict(input_data)[0]

    st.success("Predicted Genre: " + str(prediction))

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]
        confidence = max(probabilities) * 100
        st.metric("Confidence", f"{confidence:.2f}%")
