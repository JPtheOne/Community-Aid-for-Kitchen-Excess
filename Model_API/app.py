from flask import Flask, jsonify, request
import tensorflow as tf
import pandas as pd

app = Flask(__name__)

# Load the model (make sure the path is correct)
model = tf.keras.models.load_model('pModel.keras')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    prediction = model.predict(df[['feature1', 'feature2', 'feature3']])
    return jsonify({'prediction': prediction.tolist()})

if __name__ == "__main__":
    app.run(debug=True, port=5000)
