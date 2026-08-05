# Health Chatbot

A health symptom-checker chatbot. Users describe their symptoms in plain
English, and the app predicts a likely disease using a machine learning
model, then uses Google's Gemini API to explain the prediction in simple,
friendly language.

> **Disclaimer:** This is a student project for learning purposes only.
> It is **not** a substitute for professional medical advice, diagnosis,
> or treatment. Always consult a real doctor for actual health concerns.

## How it works

1. User types symptoms in free text (typos are OK)
2. `symptom_matcher.py` matches typed words to known symptom names using
   fuzzy string matching (offline, no API needed)
3. A Random Forest model (trained on `Training.csv`) predicts the most
   likely disease from the matched symptoms
4. The Gemini API explains the prediction in plain language
5. The frontend displays the result in a chat interface

## Tech stack

- **ML model:** scikit-learn (Random Forest Classifier)
- **Backend:** Flask (Python)
- **Explanation:** Google Gemini API
- **Frontend:** HTML / CSS / JavaScript

## Project structure

| File | Purpose |
|---|---|
| `main.py` | Loads data, trains the model, and defines the core prediction functions |
| `app.py` | Flask app — exposes the `/predict` API endpoint for the frontend |
| `symptom_matcher.py` | Fuzzy-matches user-typed symptoms to known symptom names |
| `gemini_helper.py` | Calls the Gemini API to generate the plain-language explanation |
| `Training.csv` | Symptom-disease dataset used to train the model |

## Setup

### 1. Get a free Gemini API key

1. Go to [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Sign in with any Google account
3. Click **Create API key**
4. Copy the key (starts with something like `AIzaSy...`)

The key is never included in this repository for security reasons — you
need your own.

### 2. Set the key on your machine

**Windows (PowerShell):**
```powershell
setx GEMINI_API_KEY "your_key_here"
```
Then **close and reopen PowerShell completely** (`setx` only applies to
new terminal windows).

Verify it worked:
```powershell
echo $env:GEMINI_API_KEY
```

### 3. Install required libraries

```bash
pip install numpy pandas scikit-learn google-genai flask flask-cors
```

## Running the project

### Backend (required)

```bash
py app.py
```

This starts the Flask server at `http://localhost:5000`. Leave this
terminal running.

### Frontend

Open the frontend's HTML file in your browser (or serve it however your
frontend setup requires). It sends requests to the backend at
`http://localhost:5000/predict`.

### Testing the backend alone (optional)

You can run `main.py` directly to test predictions from the terminal,
without the Flask server or frontend:

```bash
py main.py
```

## API Reference

### `POST /predict`

**Request body:**
```json
{
  "symptoms": ["nausea", "anxiety"]
}
```

**Response:**
```json
{
  "disease": "Hepatitis C",
  "symptoms_used": ["nausea", "anxiety"],
  "unmatched": [],
  "explanation": "..."
}
```

- `disease` — predicted disease name
- `symptoms_used` — symptoms that were successfully matched and used for prediction
- `unmatched` — symptoms typed by the user that couldn't be confidently matched
- `explanation` — Gemini's plain-language explanation of the prediction
