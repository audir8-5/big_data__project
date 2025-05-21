from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from datetime import datetime
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load("ridge_weather_model.joblib")

# Get expected city categories from the model
city_encoder = model.named_steps['preprocessor'].named_transformers_['cat']
expected_cities = city_encoder.categories_[0].tolist()

@app.route("/")
def home():
    return render_template("page.html")

@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        # Extract features from JSON
        dew_point = data["main"]["temp"] - ((100 - data["main"]["humidity"]) / 5)
        apparent_temp = data["main"]["feels_like"]
        precipitation = 0.0  # default
        rain = 0.0
        wind_direction = data["wind"]["deg"]
        timestamp = data["dt"]
        dt_obj = datetime.utcfromtimestamp(timestamp + data["timezone"])
        hour = dt_obj.hour
        month = dt_obj.month
        city = data["name"]

        # Check city
        if city not in expected_cities:
            return jsonify({"error": f"City '{city}' not in trained cities. Expected one of: {expected_cities}"}), 400

        input_data = {
            "dew_point": dew_point,
            "apparent_temperature": apparent_temp,
            "precipitation": precipitation,
            "rain": rain,
            "wind_direction": wind_direction,
            "hour": hour,
            "month": month,
            "city": city
        }

        input_df = pd.DataFrame([input_data], columns=[
            'dew_point', 'apparent_temperature', 'precipitation', 'rain',
            'wind_direction', 'hour', 'month', 'city'
        ])

        prediction = model.predict(input_df)[0]

        result = {
            "predicted_temperature": round(prediction[0], 2),
            "predicted_relative_humidity": round(prediction[1], 2),
            "predicted_wind_speed": round(prediction[2], 2)
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run()
