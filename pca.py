import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA

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

    # El número de componentes de PCA es opcional, por defecto el numero es igual al
    # minimo entre el número de muestras y el número de features.
    pca = PCA(n_components=3)

    # Ajustamos el PCA a los datos de entrenamiento
    pca.fit(X_train)

    # Implementando IPCA
    # El parámetro batch_size es para segmentar los datos al mandarlos a entrenar y no sobre exigir la capacidad de cómputo
    ipca = IncrementalPCA(n_components=3, batch_size=10)
    ipca.fit(X_train)

    # Imprimo la cantidad de componentes generados de PCA, nuestro eje X
    #plt.plot(range(len(pca.explained_variance_)),
    #        pca.explained_variance_ratio_)  # eje Y

    # El gráfico que veremos mostrará la importancia (eje y) de cada componente (eje x), como se puede ver, el primero es el que
    # aporta mayor cantidad de info con un 0.22
    plt.show()

    # Configuramos nuestra regresión logística
    # el parametro es solo convencion
    logistic = LogisticRegression(solver='lbfgs')

    # creamos dataset de entrenamiento
    ds_train = pca.transform(X_train)

    # creamos dataset de prueba
    ds_test = pca.transform(X_test)

    # Ajustando el modelo
    logistic.fit(ds_train, y_train)

    # Imprimiendo métricas para evaluar el modelo
    print('SCORE PCA: ', logistic.score(ds_test, y_test))

    # Entreno con IPCA
    ds_train = ipca.transform(X_train)
    dt_test = ipca.transform(X_test)
    logistic.fit(ds_train, y_train)
    print("SCORE IPCA: ", logistic.score(ds_test, y_test))
