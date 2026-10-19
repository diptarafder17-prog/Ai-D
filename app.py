from flask import Flask, request, jsonify
import joblib
import pandas as pd
import requests
from io import StringIO

app = Flask(__name__)
MODEL_PATH = "model.pkl"

def load_model():
    return joblib.load(MODEL_PATH)

model = None

@app.before_first_request
def startup():
    global model
    model = load_model()

@app.route("/predict", methods=["POST"])
def predict():
    """
    POST JSON body:
    {
      "source_url": "https://raw.githubusercontent.com/..../iris.csv",
      "row_index": 0
    }
    If source_url is omitted the server will fetch the default Iris CSV.
    Or you can POST {"features": [5.1, 3.5, 1.4, 0.2]} to predict a single sample.
    """
    data = request.get_json() or {}
    # direct features provided
    if "features" in data:
        features = [data["features"]]
        pred = model.predict(features)[0]
        return jsonify({"prediction": str(pred)})
    # fetch a CSV from the web and pick a row
    url = data.get("source_url", "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
    row_index = int(data.get("row_index", 0))
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
        row = df.iloc[row_index]
        features = [row[["sepal_length", "sepal_width", "petal_length", "petal_width"]].values.tolist()]
        pred = model.predict(features)[0]
        return jsonify({"prediction": str(pred), "source_row": row.to_dict()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
