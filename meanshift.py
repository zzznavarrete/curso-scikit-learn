import pandas as pd

from sklearn.cluster import MeanShift

if __name__ == "__main__":
    
    dataset = pd.read_csv("./datasets/candy.csv")
    print(dataset.head())
    
    X = dataset.drop("competitorname", axis=1)

    # El "ancho de banda" es el parámetro más importante para mean shift,
    # lo dejaremos por defecto para que lo calculo por sí mismo
    meanshift = MeanShift().fit(X)

    # viendo la cant. de clusters
    print(max(meanshift.labels_))

    # Viendo los centros
    print("="* 64)
    print(meanshift.cluster_centers_)

    dataset["meanshift"] = meanshift.labels_
    print("="*64)
    print(dataset)