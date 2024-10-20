import pandas as pd
from flask import Flask, request, jsonify
import joblib
from DataCleaning import data_cleaning

# Create Flask app
app = Flask(__name__)

# Load the pickle model
random_model = joblib.load('models\Model_Classifier_Electric.pkl')
features = joblib.load('models\Features_Columns.pkl')

# Define a route for the prediction endpoint
@app.route('/predict', methods=["POST"])

def predict():
    data = request.json
    input_data = pd.DataFrame([data['data']])  # Wrapping in a list for a single row

    X_input = data_cleaning(input_data)

    print("Processed input data for prediction:", X_input)
    target_column = 'pm'  # Replace with your actual target column name
    if target_column in X_input.columns:
        X_input = X_input.drop(columns=[target_column])

    predictions = random_model.predict(X_input)
    return jsonify({'predictions': predictions.tolist()})

# Run the app if executed directly
if __name__ == "__main__":
    app.run()
