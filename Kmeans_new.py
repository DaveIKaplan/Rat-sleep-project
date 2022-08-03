import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


if __name__ == "__main__":

	"""Read in the CSV data from the filename specified"""
	data = pd.read_csv("data/Cluster_1_2.csv")

	"""Standardise the data to have a mean of 0 and a variance of 1"""
	standard_scaler = StandardScaler()
	X_std = standard_scaler.fit_transform(data)

	"""Perform PCA on the data with 12 dimensions, plot the 
	explained variance ratio of each dimension, and create a 
	scatter plot of the first 2 dimensions"""
	dimensions = 12
	pca = PCA(n_components=dimensions)
	X_pca = pca.fit_transform(X_std)
	plt.bar(range(dimensions), pca.explained_variance_ratio_, color='black')
	plt.xlabel('PCA features')
	plt.ylabel('variance %')
	plt.xticks(range(dimensions))
	plt.show()
	print(f"PCA components size: {X_pca.shape}")
	plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=.1, color='black')
	plt.xlabel('PCA 1')
	plt.ylabel('PCA 2')
	plt.show()

	"""Run an experiment to obtain the ideal number of 
	clusters (between 1 and 10) based on the plot of
	model inertia's"""
	models = []
	num_clusters = np.arange(1, 11)
	for k in num_clusters:
		""" Create a KMeans instance with k clusters"""
		model = KMeans(n_clusters=k)
		"""Fit the model on the first 2 PCA dimensions"""
		model.fit(X_pca[:, :2])
		"""Append the model to the list of models"""
		models.append(model)

	model_inertias = [model.inertia_ for model in models]
	plt.plot(num_clusters, model_inertias, '-o', color='black')
	plt.xlabel('Number of clusters, k')
	plt.ylabel('Inertia')
	plt.xticks(num_clusters)
	plt.show()

	ideal_num_clusters = 3
	model = models[ideal_num_clusters - 1]
	for i in range(ideal_num_clusters):
		print(f'Cluster {i+1} size: {np.sum(model.labels_ == i)}')

	color_vector = [
		"green", "red", "blue", "orange", "black",
		"purple", "cyan", "yellow", "red"
	]
	colors = np.array(color_vector)[model.labels_]
	plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=1, color=colors)
	plt.xlabel('PCA 1')
	plt.ylabel('PCA 2')
	plt.show()

	print(f'PCA components type: {type(pca.components_)}')
	print(f'PCA components shape: {pca.components_.shape}')
	plt.matshow(abs(pca.components_[:2, :]), cmap='viridis')
	plt.yticks([0, 1], ['1st Comp', '2nd Comp'], fontsize=10)
	plt.colorbar()
	features = data.columns
	plt.xticks(range(len(features)), features, rotation=65, ha='left')
	plt.tight_layout()
	plt.show()

	"""Save a csv file for each cluster"""
	for i in range(ideal_num_clusters):
		df = pd.DataFrame({f"Cl{i+1}": data.iloc[model.labels_ == i, 0]})
		df.to_csv(f"data/CellName_cluster{i+1}_new.csv", index=False)
