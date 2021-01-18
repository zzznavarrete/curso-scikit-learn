import pandas as pd

# Casi el mismo algoritmo pero consume mucho menos recursos
from sklearn.cluster import MiniBatchKMeans


if __name__ == "__main__":
    dataset = pd.read_csv("./datasets/candy.csv")
    print(dataset.head())

    X = dataset.drop("competitorname", axis=1)
    kmeans = MiniBatchKMeans(n_clusters=4, batch_size=8).fit(X)
    print("total de centros: ", len(kmeans.cluster_centers_))

    print("="*64)
    print(kmeans.predict(X))

    # Integro la predicción
    dataset['group'] = kmeans.predict(X)
    print(dataset)