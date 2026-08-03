import os
from google import genai
from google.genai import errors as genai_errors


def get_gemini_client():
    
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set. "
            "Run: setx GEMINI_API_KEY \"your_key_here\" then restart your terminal."
        )
    return genai.Client(api_key=api_key)


def build_explanation_prompt(predicted_disease: str, user_symptoms: list) -> str:
    
    symptoms_text = ", ".join(s.strip() for s in user_symptoms)

    prompt = f"""
You are a medical information assistant. A machine learning model has ALREADY
predicted the following disease based on a patient's symptoms. Do NOT question,
change, or re-diagnose the prediction -- your only job is to explain it clearly.

Predicted disease: {predicted_disease}
Symptoms reported by the patient: {symptoms_text}

Please write a well-formatted, easy-to-read response with the following sections,
using clear headings:

1. Disease Overview
2. Why the Reported Symptoms May Relate to This Disease
3. Common Causes
4. Typical Symptoms
5. General Self-Care Advice
6. Foods to Eat or Avoid (if relevant to this condition)
7. When to Seek Medical Attention
8. Prevention Tips

End with this exact disclaimer on its own line:
"Disclaimer: This information is for educational purposes only and is not a
medical diagnosis. Please consult a qualified healthcare professional for
proper evaluation and treatment."
"""
    return prompt.strip()


def explain_prediction(predicted_disease: str, user_symptoms: list) -> str:

    try:
        client = get_gemini_client()
        prompt = build_explanation_prompt(predicted_disease, user_symptoms)

        response = client.models.generate_content(
            model="gemini-3.6-flash",  # swap for a newer model name if this one gets deprecated too
            contents=prompt,
        )

        if not response or not getattr(response, "text", None):
            return "Could not generate explanation: Gemini returned an empty response."

        return response.text.strip()

    except genai_errors.APIError as e:
        return f"Could not generate explanation: Gemini API error: {e}"
    except ConnectionError as e:
        return f"Could not generate explanation: network/connection error: {e}"
    except RuntimeError as e:
        return f"Could not generate explanation: {e}"
    except Exception as e:
        return f"Could not generate explanation: unexpected error: {e}"