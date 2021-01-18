import pandas as pd

# Importo los regresores
from sklearn.linear_model import (
    RANSACRegressor, HuberRegressor
)
# Comparamos el resultado con un modelo basado en soporte de maquinas vectoriales
# Importanmos el regresor, Support Vector Regressor (SVR)
from sklearn.svm import SVR

# para separar el dataset
from sklearn.model_selection import train_test_split

# metricas
from sklearn.metrics import mean_squared_error


if __name__ == "__main__":
    dataset = pd.read_csv('./datasets/felicidad_corrupt.csv')
    print(dataset.head())

    # Definiendo nuestros features

    X = dataset.drop(['country', 'score'], axis=1)
    y = dataset[['score']]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=31)

    # Como trabajaremos con varios estimadores, codificaremos de la siguiente manera:
    # diccionario
    estimadores = {
        'SVR': SVR(gamma='auto', C=1.0, epsilon=0.1),
        'RANSAC': RANSACRegressor(),
        'HUBER': HuberRegressor(epsilon=1.35)
    }

    # Entreno los modelos por su tupla en el diccionario
    for name, estimador in estimadores.items():

        # Ajusto y entreno
        estimador.fit(X_train, y_train)
        predictions = estimador.predict(X_test)

        # Imprimo las metricas
        # vemos que el valor de las salidas de las regresiones robustas son mucho menores que a
        # a nuestro modelo de soporte vectorial el cual toma en cuenta los outliers a diferencia de las robustas.
        print('='*64)
        print(name)
        print("MSE = ", mean_squared_error(y_test, predictions))
