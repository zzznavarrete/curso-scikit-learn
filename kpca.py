import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.decomposition import KernelPCA


from sklearn.linear_model import LogisticRegression

# Utils
# Permite normalizar los datos para que estén en escala de 0 a 1
from sklearn.preprocessing import StandardScaler

# divide los datos para pruebas y entrenamiento
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    # Cargo dataset
    # Trabajo sobre el dataset de pacientes con riesgo de enfermedad cardiaca.
    # Utilizando ciertos parametros haremos una clasificacion binaria entre
    # Si el paciente tiene o no una enfermedad cardiaca.
    ds_heart = pd.read_csv('./datasets/heart.csv')
    print(ds_heart.head())
    # parto mi dataset entre feature y target.

    # Guardo primero todos los features en un dataset
    ds_features = ds_heart.drop(['target'], axis=1)

    # Guardo mi dataset de target
    ds_target = ds_heart['target']

    # Normalizo mis datos, escala nuestros datos
    ds_features = StandardScaler().fit_transform(ds_features)
    # El target sigue igual, no se toca.

    #Separo los datasets
    # Los parámetros son, primero los 2 dataset, luego el % de lo que quiero que sea para test, y finalmente ->
    # un número de random state para asegurar que siempre que ejecute la linea sean los mismos resultados y no cambie el orden o el sorteo del dataset.
    X_train, X_test, y_train, y_test = train_test_split(
        ds_features, ds_target, test_size=0.3, random_state=31)

    # Vemos los datos consultando la forma con el parámetro .shape
    print(X_train.shape)
    print(y_train.shape)

    # Aplicamos un KernelPCA. (Ver docs para mayor info)
    kpca = KernelPCA(n_components=4, kernel="poly")
    # Ajusto los datos
    kpca.fit(X_train)

    # aplicamos sobre datos de entrenamiento y de prueba
    ds_train = kpca.transform(X_train)
    ds_test = kpca.transform(X_test)

    # Aplicamos nuevamente nuestra regresión logística
    logistic = LogisticRegression(solver='lbfgs')
    # entreno el modelo
    logistic.fit(ds_train, y_train)

    print('SCORE KPCA: ', logistic.score(ds_test, y_test))
