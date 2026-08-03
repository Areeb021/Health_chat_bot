import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# ----------------- ADDED (new block) -----------------------------------------------
from gemini_helper import explain_prediction
from symptom_matcher import correct_symptom_names
# ^ CHANGED: correct_symptom_names now comes from the new offline
#   symptom_matcher.py (uses difflib, no internet/API needed) instead of
#   gemini_helper.py. Everything else below still works exactly the same,
#   since the function name and return format are identical.
# ------------------ END ADDED (new block) -------------------------------------------

df = pd.read_csv("Training.csv")

# print(df.isnull().sum())
df = df.drop(['Unnamed: 133'], axis=1)
# print(df.columns)

le = LabelEncoder()
encoded_labels = le.fit_transform(df['prognosis'])
print(encoded_labels)

X = df.drop(['prognosis'], axis=1)
y = encoded_labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# scaler=StandardScaler()
# X_scaled_train=scaler.fit_transform(X_train)
# X_scaled_test=scaler.fit(X_test)
rf = RandomForestClassifier(random_state=42, n_estimators=100)
rf.fit(X_train, y_train)
# pred=rf.predict(X_test)
# print(pred)
# print(le.classes_)

zero_array = np.zeros(132).reshape(1, 132)
# print(zero_array.shape)
list_diseases = list(df.columns.drop(['prognosis']))
# print(list_disease)

# --------------------------- ADDED (new block) ------------------------------------
# Instead of forcing the user to pick exact numbers, let them type symptoms
# in their own words (typos are OK). correct_symptom_names() uses offline
# string-similarity matching (difflib) to find the closest real symptom
# name for each word typed -- no internet or API call needed.
print("\nYou can describe your symptoms in your own words (typos are OK).")
free_text_input = input("Enter your symptoms, comma-separated: ")
typed_symptoms = free_text_input.split(",")

correction_result = correct_symptom_names(typed_symptoms, list_diseases)
symptom_names = correction_result["matched"]
unmatched = correction_result["unmatched"]

if unmatched:
    print(f"\nCould not confidently match these: {unmatched}")

# --- FALLBACK ---
# If nothing matched closely enough (e.g. the typed word was too different
# from any real symptom), fall back to the numbered-list method so the
# user can still continue.
if not symptom_names:
    print("\nFalling back to manual selection.")
    print("Available symptoms:")
    for i, symptom in enumerate(list_diseases):
        print(f"{i}: {symptom}")
    u_input = input("\nEnter the symptom NUMBERS you have (comma-separated): ")
    u_input = u_input.split(",")
    for symptoms in u_input:
        idx = int(symptoms.strip())
        zero_array[0][idx] = 1
        symptom_names.append(list_diseases[idx])
else:
    # Mark each matched symptom as present in the feature vector
    for name in symptom_names:
        idx = list_diseases.index(name)
        zero_array[0][idx] = 1

u_input = symptom_names
# ^ u_input now holds the final, clean symptom NAMES, ready to be used for
#   prediction and passed to Gemini for the explanation step below.
# ------------------------------- END ADDED BLOCK -----------------------------------

print(zero_array)

# getting prediction

pred = rf.predict(zero_array)

predicted_disese = le.inverse_transform(pred)
print(predicted_disese)

# --- ADDED (new block) ---
# Your Random Forest model has ALREADY made the prediction above
# (predicted_disese). This block does NOT re-predict anything -- it just
# sends the finished result to Gemini so it can explain it in plain English.
#
# explain_prediction() lives in gemini_helper.py and takes 2 arguments:
#   1. predicted_disese[0] -> the disease name your model predicted
#      (it's [0] because inverse_transform() returns an array, and we
#       only need the single value inside it)
#   2. u_input -> the list of symptom NAMES the user selected (converted
#      from numbers earlier in this file)
#
# It returns a string: either Gemini's explanation, OR (if something goes
# wrong, like a bad API key or no internet) a readable error message --
# it never crashes the program.
explanation = explain_prediction(predicted_disese[0], u_input)

print("\n" + "=" * 70)
print("AI-GENERATED EXPLANATION")
print("=" * 70)
print(explanation)
# --- END ADDED BLOCK ---