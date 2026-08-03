# Gemini API Key Setup

This project uses Google's Gemini API to generate explanations for the
disease predictions. To run the project, you need your **own** free API key
-- the key is never included in this repository for security reasons.

## 1. Get a free API key

1. Go to [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Sign in with any Google account
3. Click **Create API key**
4. Copy the key (starts with something like `AIzaSy...`)

## 2. Set the key on your machine

### Method Environment Variable 

**Windows (PowerShell):**
```powershell
setx GEMINI_API_KEY "your_key_here"
```
Then **close and reopen PowerShell completely** (setx only applies to new
terminal windows).

Verify it worked:
```powershell
echo $env:GEMINI_API_KEY
```

## 3. Install required libraries

```bash
pip install numpy pandas scikit-learn google-genai
```

## 4. Run the project

```bash
python main.py
```

