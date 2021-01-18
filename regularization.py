import pandas as pd
import sklearn

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


if __name__ == "__main__":
    # Cargo el dataset
    dataset = pd.read_csv('./datasets/felicidad.csv')
    # Imprimo el summario para testear si se carg'o correctamente
    print(dataset.describe())

    # dividimos el dataset en features y target
    X = dataset[['gdp',  'family', 'lifexp', 'freedom',
                 'corruption', 'generosity', 'dystopia']]
    y = dataset[['score']]

    # Comprobando que esten correctamente
    print(X.shape)
    print(y.shape)

    # hago el split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # ajusto el modelo
    model_linear = LinearRegression().fit(X_train, y_train)
    y_predict_linear = model_linear.predict(X_test)

    # Ajusto el modelo lasso, mientras mas grande dejo el alpha, m'as penalizacion tiene
    model_lasso = Lasso(alpha=0.02).fit(X_train, y_train)
    y_predict_lasso = model_lasso.predict(X_test)

    # entreno ridge
    model_ridge = Ridge(alpha=1).fit(X_train, y_train)
    y_predict_ridge = model_ridge.predict(X_test)

    # Examino perdidas
    # Mientras menor perdida es mejor, sigfnifica que hay menor error entre los valores
    # esperados y los valores predichos.
    linear_loss = mean_squared_error(y_test, y_predict_linear)
    print("Linear Loss: ", linear_loss)

    lasso_loss = mean_squared_error(y_test, y_predict_lasso)
    print("Lasso Loss: ", lasso_loss)

    ridge_loss = mean_squared_error(y_test, y_predict_ridge)
    print("Ridge Loss: ", ridge_loss)

    # Veo los coeficientes
    # Los coeficientes es el valor con el cual opera nuestros features,
    # por la naturaleza de Lasso, algunos ser'an 0  por lo que no afectar'an al modelo
    # En cambio en Ridge, ser'an muy cercanos a 0 pero no 0.
    print("="*32)
    print("Coef LASSO")
    print(model_lasso.coef_)
    print("="*32)
    print("Coef RIDGE")
    print(model_ridge.coef_)
