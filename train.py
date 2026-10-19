import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Download a small public dataset (Iris) from the web
URL = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"

def main():
    df = pd.read_csv(URL)
    # features: sepal_length, sepal_width, petal_length, petal_width
    X = df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
    y = df["species"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, preds))
    joblib.dump(clf, "model.pkl")
    print("Saved model to model.pkl")

if __name__ == "__main__":
    main()
