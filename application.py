from flask import Flask, render_template, request
import pandas as pd 
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

with open("models/student_performance_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form values
        hours = float(request.form['hours'])
        scores = float(request.form['scores'])
        activities = int(request.form['activities'])
        sleep = float(request.form['sleep'])
        papers = int(request.form['papers'])

        # Prepare input for model
        input_data=scaler.transform([[hours, scores, activities, sleep, papers]])
        prediction = model.predict(input_data)[0]

        # Clamp value to 0-100
        performance_index = round(min(100, max(0, prediction)), 2)

        return render_template("index.html", result=performance_index)

    except Exception as e:
        return f"Error occurred: {e}", 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
