import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from gemini_helper import explain_prediction
from symptom_matcher import correct_symptom_names

# ============================================================
# STARTUP CODE — runs ONCE when the file is imported/run.
# Nothing here depends on any individual user's input.
# ============================================================
df = pd.read_csv("Training.csv")
df = df.drop(['Unnamed: 133'], axis=1)

le = LabelEncoder()
encoded_labels = le.fit_transform(df['prognosis'])

X = df.drop(['prognosis'], axis=1)
y = encoded_labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestClassifier(random_state=42, n_estimators=100)
rf.fit(X_train, y_train)

list_diseases = list(df.columns.drop(['prognosis']))


# ============================================================
# FUNCTION 1 — turn a list of typed symptoms into a feature vector.
# Takes symptoms as a PARAMETER (list of strings) — no input() here.
# This is what makes it Flask-safe: the caller decides where the
# symptoms come from (terminal for testing, JSON request for Flask).
# ============================================================
def get_symptom_vector(typed_symptoms):
    zero_array = np.zeros(132).reshape(1, 132)  # fresh array every call — fixes the "leftover 1s" bug

    correction_result = correct_symptom_names(typed_symptoms, list_diseases)
    symptom_names = correction_result["matched"]
    unmatched = correction_result["unmatched"]

    if not symptom_names:
        # Fallback path — caller (terminal test or Flask route) handles
        # asking the user again; this function just reports nothing matched.
        return zero_array, [], unmatched

    for name in symptom_names:
        idx = list_diseases.index(name)
        zero_array[0][idx] = 1

    return zero_array, symptom_names, unmatched


# ============================================================
# FUNCTION 2 — run the trained model on a feature vector.
# ============================================================
def predict_disease(zero_array):
    pred = rf.predict(zero_array)
    predicted_disease = le.inverse_transform(pred)
    return predicted_disease


# ============================================================
# FUNCTION 3 — get Gemini's plain-English explanation.
# ============================================================
def get_explanation(predicted_disease, symptom_names):
    return explain_prediction(predicted_disease[0], symptom_names)


# ============================================================
# TERMINAL TEST — only runs when you do `python main.py` directly.
# Flask will import this file's functions WITHOUT running this block.
# ============================================================
if __name__ == "__main__":
    print("\nYou can describe your symptoms in your own words (typos are OK).")
    free_text_input = input("Enter your symptoms, comma-separated: ")
    typed_symptoms = free_text_input.split(",")

    zero_array, symptom_names, unmatched = get_symptom_vector(typed_symptoms)

    if unmatched:
        print(f"\nCould not confidently match these: {unmatched}")

    if not symptom_names:
        print("\nNo symptoms matched — try different wording.")
    else:
        predicted_disease = predict_disease(zero_array)
        print(predicted_disease)

        explanation = get_explanation(predicted_disease, symptom_names)
        print("\n" + "=" * 70)
        print("AI-GENERATED EXPLANATION")
        print("=" * 70)
        print(explanation)