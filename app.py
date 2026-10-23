
import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open("svc_trained_model.pkl", "rb") as file:
    model = pickle.load(file)

# Page settings
st.set_page_config(page_title="Titanic Survival Prediction")
st.title("Titanic Passenger Survival Prediction")
st.write("Enter passenger details to predict survival.")

# Input fields
pclass = st.selectbox(
    "Passenger Class",
    ["First", "Second", "Third"]
)

gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

sibling = st.selectbox(
    "Number of Siblings",
    ["Zero", "One", "Two", "Three"]
)

embarked = st.selectbox(
    "Port of Embarkation",
    ["Cherbourg", "Queenstown", "Southampton"]
)

# Convert categories into the same numbers used during training
pclass_map = {"First": 0, "Second": 1, "Third": 2}
gender_map = {"Female": 0, "Male": 1}
sibling_map = {"One": 0, "Three": 1, "Two": 2, "Zero": 3}
embarked_map = {
    "Cherbourg": 0,
    "Queenstown": 1,
    "Southampton": 2
}

# Predict when the button is clicked
if st.button("Predict Survival"):
    passenger = pd.DataFrame(
        [[
            pclass_map[pclass],
            gender_map[gender],
            sibling_map[sibling],
            embarked_map[embarked]
        ]],
        columns=["PClass", "Gender", "Sibling", "Embarked"]
    )

    prediction = model.predict(passenger)[0]

    if prediction == 1:
        st.success("Prediction: SURVIVED")
    else:
        st.error("Prediction: NOT SURVIVED")
