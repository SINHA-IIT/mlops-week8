import pandas as pd
import joblib
from sklearn.metrics import accuracy_score,precision_score, recall_score, f1_score, classification_report

def main():
    df = pd.read_csv('data/data_iris.csv')
    X = df[['sepal_length','sepal_width','petal_length','petal_width']]
    y = df['species']
    model = joblib.load('model/model.pkl')
    preds = model.predict(X)
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds, average='weighted')
    rec = recall_score(y, preds, average='weighted')
    f1 = f1_score(y, preds, average='weighted')

    metrics = {
        'accuracy': [acc],
        'precision': [prec],
        'recall': [rec],
        'f1_score': [f1]
    }

    # Create metrics.csv
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv('metrics.csv', index=False)
    print(metrics_df)
    print("Accuracy:", acc)
    print(classification_report(y, preds))

if __name__ == "__main__":
    main()
