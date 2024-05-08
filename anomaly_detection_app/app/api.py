from flask import Flask, request, jsonify
from .models import Model

app = Flask(__name__)
model = Model('conf.json')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    predictions = model.predict(data['features'])
    return jsonify(predictions.tolist())
