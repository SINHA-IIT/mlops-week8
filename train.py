import os
import sys
import pandas as pd
import numpy as np
import joblib
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
import matplotlib.pyplot as plt

def log_confusion_matrix(cm, labels, out_path):
    plt.figure(figsize=(4,4))
    plt.imshow(cm, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks(np.arange(len(labels)), labels, rotation=45)
    plt.yticks(np.arange(len(labels)), labels)

    for i in range(len(cm)):
        for j in range(len(cm)):
            plt.text(j, i, cm[i][j], ha='center', va='center', color='black')

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def main():

    poison_path = "data/data_iris_poisoned.csv"
    clean_path = "data/data_iris.csv"

    # MLflow experiment
    mlflow.set_experiment("iris_feature_poisoning")

    # Determine which dataset is being used
    if os.path.exists(poison_path):
        df = pd.read_csv(poison_path)
        poison_level = os.environ.get("POISON_LEVEL", "unknown")
        run_name = f"feature_poison_p{poison_level}"
    else:
        df = pd.read_csv(clean_path)
        poison_level = "0.0"
        run_name = "clean_data_model"

    with mlflow.start_run(run_name=run_name):

        mlflow.log_param("poison_level", poison_level)

        train, test = train_test_split(df, test_size=0.4, stratify=df['species'], random_state=42)

        X_train = train[['sepal_length','sepal_width','petal_length','petal_width']]
        y_train = train.species
        X_test = test[['sepal_length','sepal_width','petal_length','petal_width']]
        y_test = test.species

        model = DecisionTreeClassifier(max_depth=3, random_state=1)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average='weighted')
        rec = recall_score(y_test, preds, average='weighted')
        f1 = f1_score(y_test, preds, average='weighted')

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # Confusion matrix logging
        cm = confusion_matrix(y_test, preds)
        cm_path = "confusion_matrix.png"
        log_confusion_matrix(cm, df["species"].unique(), cm_path)
        mlflow.log_artifact(cm_path)

        # Save model
        joblib.dump(model, "model/model.pkl")
        mlflow.sklearn.log_model(model, "decision_tree_model")

        print("Model logged to MLflow.")
        print("Accuracy:", acc)
        print("F1 Score:", f1)

if __name__ == "__main__":
    main()
