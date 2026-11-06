import numpy as np
import pandas as pd
import streamlit as st

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = "PCOS_data.csv"
RANDOM_STATE = 42
TARGET_RECALL = 0.80   # how many PCOS cases the quiz should catch

QUIZ_COLS = ["Age (yrs)", "Weight (Kg)", "Height(Cm)", "BMI",
             "Cycle(R/I)", "Cycle length(days)",
             "Weight gain(Y/N)", "hair growth(Y/N)", "Skin darkening (Y/N)", "Hair loss(Y/N)", "Pimples(Y/N)",
             "Fast food (Y/N)", "Reg.Exercise(Y/N)",
             "Waist(inch)", "Hip(inch)", "Waist:Hip Ratio"]


@st.cache_resource(show_spinner="Training the model (first run only)...")
def train():
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip().str.replace(r"\s+", " ", regex=True)
    df = df.rename(columns={"PCOS (Y/N)": "PCOS"}).apply(pd.to_numeric, errors="coerce")
    X, y = df[QUIZ_COLS], df["PCOS"]

    def make():
        return Pipeline([("impute", SimpleImputer(strategy="median")),
                         ("scale", StandardScaler()),
                         ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                        random_state=RANDOM_STATE))])

    # Choose the flagging threshold from out-of-fold predictions
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    proba = cross_val_predict(make(), X, y, cv=cv, method="predict_proba")[:, 1]
    threshold = 0.5
    for t in np.arange(0.60, 0.10, -0.05):
        if recall_score(y, (proba >= t).astype(int)) >= TARGET_RECALL:
            threshold = float(t)
            break

    model = make().fit(X, y)
    return model, threshold


model, threshold = train()

st.title("PCOS Likelihood Quiz")
st.caption("A coursework project. This is not a medical diagnosis.")

with st.form("quiz"):
    st.subheader("About you")
    c1, c2 = st.columns(2)
    age = c1.number_input("Age (years)", 15, 60, 28)
    weight = c2.number_input("Weight (kg)", 30.0, 200.0, 60.0, step=0.5)
    height = c1.number_input("Height (cm)", 120.0, 210.0, 162.0, step=0.5)
    waist_cm = c2.number_input("Waist (cm)", 50.0, 150.0, 80.0, step=0.5)
    hip_cm = c1.number_input("Hip (cm)", 60.0, 160.0, 95.0, step=0.5)

    st.subheader("Your cycle")
    regular = st.radio("Is your cycle regular?", ["Yes", "No"], horizontal=True)
    period_days = st.slider("How many days does your period usually last?", 1, 12, 5)

    st.subheader("Symptoms")
    yn = ["No", "Yes"]
    weight_gain = st.radio("Recent weight gain?", yn, horizontal=True)
    hair_growth = st.radio("Excess hair growth (face, chest, back)?", yn, horizontal=True)
    skin_dark = st.radio("Skin darkening (neck, underarms, groin)?", yn, horizontal=True)
    hair_loss = st.radio("Hair loss or thinning?", yn, horizontal=True)
    pimples = st.radio("Frequent pimples or acne?", yn, horizontal=True)

    st.subheader("Lifestyle")
    fast_food = st.radio("Do you eat fast food regularly?", yn, horizontal=True)
    exercise = st.radio("Do you exercise regularly?", yn, horizontal=True)

    submitted = st.form_submit_button("See my result")

if submitted:
    bmi = weight / (height / 100) ** 2
    waist_in, hip_in = waist_cm / 2.54, hip_cm / 2.54
    row = pd.DataFrame([{
        "Age (yrs)": age, "Weight (Kg)": weight, "Height(Cm)": height, "BMI": bmi,
        "Cycle(R/I)": 2 if regular == "Yes" else 4,   # dataset coding: 2 = regular, 4 = irregular
        "Cycle length(days)": period_days,
        "Weight gain(Y/N)": yn.index(weight_gain), "hair growth(Y/N)": yn.index(hair_growth),
        "Skin darkening (Y/N)": yn.index(skin_dark), "Hair loss(Y/N)": yn.index(hair_loss),
        "Pimples(Y/N)": yn.index(pimples), "Fast food (Y/N)": yn.index(fast_food),
        "Reg.Exercise(Y/N)": yn.index(exercise),
        "Waist(inch)": waist_in, "Hip(inch)": hip_in, "Waist:Hip Ratio": waist_in / hip_in,
    }])[QUIZ_COLS]

    score = model.predict_proba(row)[0, 1]
    st.divider()
    if score >= threshold:
        st.warning("**Higher likelihood.** Your answers look similar to people who have PCOS. "
                   "It is worth talking to a doctor.")
    else:
        st.success("**Lower likelihood.** Your answers look less similar to people who have PCOS. "
                   "If you have symptoms that worry you, a doctor can still check.")
    st.caption("This quiz only compares your answers with a small dataset (about 540 patients). "
               "It cannot diagnose PCOS, and some people flagged will not have it.")
