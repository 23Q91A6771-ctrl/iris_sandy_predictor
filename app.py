from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

# load the saved model
model_path = os.path.join(os.getcwd(), "models", "iris_model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # get data from form
    sl = float(request.form["sepal_length"])
    sw = float(request.form["sepal_width"])
    pl = float(request.form["petal_length"])
    pw = float(request.form["petal_width"])

    # prepare feature array
    features = np.array([[sl, sw, pl, pw]])

    # make prediction
    pred = model.predict(features)[0]

    # map numeric class to label
    classes = ["Setosa", "Versicolor", "Virginica"]
    result = classes[pred]

    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)