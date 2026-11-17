
from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
saved_data = joblib.load("model.pkl")

model = saved_data["model"]
features = saved_data["features"]
model_name = saved_data.get("name", "AQI Model")


# Function to classify AQI
def get_category(aqi):

    if aqi <= 50:
        return "Good"

    elif aqi <= 100:
        return "Satisfactory"

    elif aqi <= 200:
        return "Moderate"

    elif aqi <= 300:
        return "Poor"

    elif aqi <= 400:
        return "Very Poor"

    else:
        return "Severe"


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    category = None
    error = None

    if request.method == "POST":

        try:
            values = []

            for feature in features:
                value = float(request.form[feature])

                if not np.isfinite(value) or value < 0:
                    raise ValueError("Enter valid non-negative values.")

                values.append(value)

            input_data = np.array(values).reshape(1, -1)

            result = model.predict(input_data)[0]

            result = max(0, min(500, result))

            prediction = round(float(result), 2)

            category = get_category(prediction)

        except ValueError as e:
            error = str(e)

        except Exception:
            error = "Prediction failed. Please check your model file."

    return render_template(
        "index.html",
        features=features,
        prediction=prediction,
        category=category,
        error=error,
        model_name=model_name
    )


if __name__ == "__main__":
    app.run(debug=True)
