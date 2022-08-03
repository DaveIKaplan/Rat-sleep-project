from kmeans import Experiment

if __name__ == "__main__":
    source_data_location = "data/Cluster_1_2.csv"
    experiment = Experiment(
        source_data_location=source_data_location,
        n_components=12,
        max_num_clusters=10,
        pca_dimensions=4
    )
    experiment.run()
