# PCOS Likelihood Quiz

A personal data classification project. It estimates the likelihood of PCOS (polycystic ovary syndrome) from answers a person can give at home, with no blood tests, ultrasound or clinic measurements.

**This is a personal project, not a medical tool. It cannot diagnose PCOS. If you are worried about your health, speak to a doctor.**

Live quiz: [https://pcosclassificationandquiz-gzmgksneogiegappjubqkru.streamlit.app/]

## How it works

- A Random Forest model is trained on 541 patients using 16 quiz-friendly features: age, weight, height, BMI, waist and hip, cycle regularity, period length, five symptoms, and two lifestyle questions.
- The quiz flags "higher likelihood" when the model's score passes a threshold chosen to catch about 80% of PCOS cases in cross-validation. Lowering the threshold catches more cases but gives more false alarms.
- The quiz shows a higher or lower likelihood message only. It does not show a percentage, because the score is not a calibrated risk.

## Files

| File | What it is |
| --- | --- |
| `app.py` | The Streamlit quiz app (trains the model on first run) |
| `PCOS_data.csv` | The dataset (see Data source) |
| `requirements.txt` | Python packages needed |
| `PCOS_quiz_model.ipynb` | Notebook with the full analysis: cleaning, model comparison, threshold choice, final test |

## Run it locally

```
pip install -r requirements.txt
streamlit run app.py
```

Then open the link shown in the terminal. It also works in a phone browser on the same wifi.

## Limitations

- Only about 540 patients from one source, and 36 PCOS cases in the test set, so results can swing between splits.
- Roughly 1 in 3 people flagged may not have PCOS.
- The dataset gives little detail on how patients were selected, so the model may not generalise to other groups.

## Data source

Kottarathil, P. (2020). *Polycystic ovary syndrome (PCOS)* [Data set]. Kaggle. https://www.kaggle.com/prasoonkottarathil/polycystic-ovary-syndrome-pcos
