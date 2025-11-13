import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

def test_model_accuracy():
    model = joblib.load("model/model.pkl")
    df = pd.read_csv("data/data_iris.csv")
    X = df[['sepal_length','sepal_width','petal_length','petal_width']]
    y = df['species']
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    assert acc > 0.9
