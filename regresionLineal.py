import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuración del Modelo ---
TARGET_COLUMN = 'precio'

def entrenamiento_regresion_lineal(X_train, y_train):

    try:

        # Inicializar y entrenar modelo
        modelo = LinearRegression()
        modelo.fit(X_train, y_train)

        return modelo

    except Exception as e:
        raise Exception(f"Error durante el entrenamiento: {e}")


def predict(modelo_rl, X_train,
             #Y_train, 
             X_test, 
             #Y_test
             ):
    # Predicciones
    Yr_train = modelo_rl.predict(X_train)
    Yr_test = modelo_rl.predict(X_test)

    """EG_train, EG_test, MAE_train, MAE_test, RMSE_train, RMSE_test, """

    return Yr_train, Yr_test

def metricas(Y,Yr):
    
    EG = r2_score(Y, Yr)
    MAE = mean_absolute_error(Y, Yr)
    RMSE = np.sqrt(mean_squared_error(Y, Yr))

    return EG, MAE, RMSE

    """# Métricas
    EG_train = r2_score(Y_train, Yr_train)
    EG_test = r2_score(Y_test, Yr_test)

    MAE_train = mean_absolute_error(Y_train, Yr_train)
    MAE_test = mean_absolute_error(Y_test, Yr_test)

    RMSE_train = np.sqrt(mean_squared_error(Y_train, Yr_train))
    RMSE_test = np.sqrt(mean_squared_error(Y_test, Yr_test))"""

