from io import BytesIO

import joblib
import pandas as pd

model = joblib.load("models/kmeans_model.pkl")


def label_new_data(data):
    clusters_labels = model.predict(pd.read_csv(BytesIO(data)))
    return {"cluster_labels": clusters_labels.tolist()}