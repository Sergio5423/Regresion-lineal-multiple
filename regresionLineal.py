import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# --- Configuración del Modelo ---
TARGET_COLUMN = 'precio'

"""
def get_preprocessor():    
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), NUMERICAL_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES)
        ],
        remainder='passthrough'
    )

def load_data(filepath):    
    try:
        df = pd.read_csv(filepath)
        # Asegúrate de que todas las columnas necesarias existan
        required_cols = [TARGET_COLUMN] + NUMERICAL_FEATURES + CATEGORICAL_FEATURES
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Columna requerida '{col}' no encontrada en el dataset.")
        return df
    except Exception as e:
        raise Exception(f"Error al cargar/validar el dataset: {e}")
"""

def entrenamiento_regresion_lineal(df_entradas, salida):

    try:
        # División de datos
        X_train, X_test, y_train, y_test = train_test_split(
            df_entradas, salida, test_size=0.2, random_state=42
        )

        # Inicializar y entrenar modelo
        modelo = LinearRegression()
        modelo.fit(X_train, y_train)

        # Predicciones
        y_pred_train = modelo.predict(X_train)
        y_pred_test = modelo.predict(X_test)

        # Métricas
        mse_train = mean_squared_error(y_train, y_pred_train)
        mse_test = mean_squared_error(y_test, y_pred_test)
        mae_train = mean_absolute_error(y_train, y_pred_train)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)

        metrics = {
            "modelo": modelo,
            "r2_train": r2_train,
            "rmse_train": np.sqrt(mse_train),
            "r2_test": r2_test,
            "rmse_test": np.sqrt(mse_test),          
            "mae_train": mae_train,
            "mae_test": mae_test,            
            "X_train": X_train,
            "y_train": y_train,
            "Yr_train": y_pred_train,
            "X_test": X_test,
            "y_test": y_test,
            "Yr_test": y_pred_test
        }

        return metrics

    except Exception as e:
        raise Exception(f"Error durante el entrenamiento: {e}")


def predict_new_data(model, data_dict):
    #Realiza una predicción con un nuevo conjunto de datos.
    new_data = pd.DataFrame([data_dict])
    return model.predict(new_data)[0]
"""
def get_unique_values(df):
    #Retorna los valores únicos de las características categóricas.
    unique_values = {}
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            unique_values[col] = df[col].unique().tolist()
    return unique_values
"""
if __name__ == '__main__':
    # Este bloque solo se ejecuta si corres regresionLineal.py directamente
    # No es necesario para el funcionamiento de la interfaz
    print("El archivo regresionLineal.py está listo para ser importado.")