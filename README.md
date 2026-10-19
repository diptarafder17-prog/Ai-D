# Simple Internet-connected AI service
#Ai-D

This example trains a simple RandomForest model on the Iris dataset (downloaded from the web) and exposes a Flask endpoint to get predictions. The service demonstrates a model that fetches input data from the internet before predicting.

Quick start:

1. Create a virtual environment and install deps:
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt

2. Train the model (downloads dataset from the web):
   python train.py

3. Run the server:
   python app.py

4. Test prediction (fetches the Iris CSV and predicts the first row):
   curl -X POST -H "Content-Type: application/json" -d '{}' http://127.0.0.1:5000/predict

5. Predict specific features:
   curl -X POST -H "Content-Type: application/json" -d '{"features":[5.1,3.5,1.4,0.2]}' http://127.0.0.1:5000/predict

Notes:
- This example is intentionally simple. For production use you must add input validation, authentication, logging, and safety checks before fetching arbitrary URLs.
- If you want the service to call external APIs (e.g., to fetch live data), you can modify the code to call those APIs and use the responses as features.
