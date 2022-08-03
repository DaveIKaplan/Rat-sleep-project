import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import joblib


COLOR_VECTOR = [
    "green", "red", "blue", "orange", "black",
    "purple", "cyan", "yellow", "red"
]


class PCADimensionSelector(BaseEstimator, TransformerMixin):
    def __init__(self, pca_dimensions):
        self.pca_dimensions = pca_dimensions

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[:, :self.pca_dimensions]


class Experiment:

    def __init__(self, source_data_location, n_components=12, max_num_clusters=10, pca_dimensions=4):
        self.source_data_location = source_data_location
        self.n_components = n_components
        self.max_num_clusters = max_num_clusters
        self.pca_dimensions = pca_dimensions

    def run(self):
        self._load_data()
        self._get_pipeline()
        self._run_experiment()
        self._select_champion_model()
        self._save_champion_model()

    def _load_data(self):
        self.X = pd.read_csv(self.source_data_location)

    def _get_pipeline(self):
        self.pipeline = make_pipeline(
            StandardScaler(),
            PCA(n_components=self.n_components),
            PCADimensionSelector(pca_dimensions=self.pca_dimensions)
        )

    def _run_experiment(self):
        self.models = []
        self.num_clusters = np.arange(1, self.max_num_clusters + 1)
        self.X_transformed = self.pipeline.fit_transform(self.X.copy())
        for k in self.num_clusters:
            """ Create a KMeans instance with k clusters"""
            model = KMeans(n_clusters=k)
            """Fit the model on the first n PCA dimensions"""
            model.fit(self.X_transformed)
            """Append the model to the list of models"""
            self.models.append(model)

    def _select_champion_model(self):
        while True:
            model_inertias = [model.inertia_ for model in self.models]
            plt.plot(self.num_clusters, model_inertias, '-o', color='black')
            plt.xlabel('Number of clusters, k')
            plt.ylabel('Inertia')
            plt.xticks(self.num_clusters)
            plt.show()

            ideal_num_clusters = int(input("Ideal Num Clusters?"))
            self.model = self.models[ideal_num_clusters - 1]

            for i in range(ideal_num_clusters):
                print(f'Cluster {i + 1} size: {np.sum(self.model.labels_ == i)}')
            colors = np.array(COLOR_VECTOR)[self.model.labels_]

            plt.scatter(self.X_transformed[:, 0], self.X_transformed[:, 1], alpha=1, color=colors)
            plt.xlabel('PCA 1')
            plt.ylabel('PCA 2')
            plt.show()

            model_confirmation = input("Are you happy with this model? (y/n)")
            if model_confirmation == "y":
                break
            elif model_confirmation == "n":
                continue

    def _save_champion_model(self):
        self.pipeline.steps.append(["kmeans", self.model])
        joblib.dump(self.pipeline, 'models/kmeans_model.pkl')
        print("Successfully saved champion model!")