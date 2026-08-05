from flask import Flask, request, jsonify
from flask_cors import CORS

from main import get_symptom_vector, predict_disease, get_explanation

app = Flask(__name__)
CORS(app)  # allows frontend (different port) to call this API

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    typed_symptoms = data['symptoms']  # expects {"symptoms": ["nausea", "anxiety"]}

    zero_array, symptom_names, unmatched = get_symptom_vector(typed_symptoms)

    if not symptom_names:
        return jsonify({'error': 'No symptoms matched', 'unmatched': unmatched}), 400

    predicted_disease = predict_disease(zero_array)
    explanation = get_explanation(predicted_disease, symptom_names)

    return jsonify({
        'disease': predicted_disease[0],
        'symptoms_used': symptom_names,
        'unmatched': unmatched,
        'explanation': explanation
    })

if __name__ == '__main__':
    app.run(debug=True ,use_reloader=False)