from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model
obj = joblib.load("california.joblib")

model = obj["Model"]
columns = obj["Columns"]

print("Model columns:", columns)


@app.route("/")
def home():
    return render_template("index.html", columns=columns)


@app.route("/predict", methods=["POST"])
def predict():

    input_data = []

    try:
        for column in columns:
            value = float(request.form[column])
            input_data.append(value)

        # Convert input into numpy array
        input_array = np.array([input_data])

        # Prediction
        prediction = model.predict(input_array)[0]

        return render_template(
            "index.html",
            columns=columns,
            prediction=round(float(prediction), 2),
            values=request.form
        )

    except Exception as e:

        return render_template(
            "index.html",
            columns=columns,
            error=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    app.run(debug=True)