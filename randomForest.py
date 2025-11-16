import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def entrenar(X_train, Y_train, n_estimators=200, max_depth=12, random_state=42, n_jobs=-1): 
    modelo_rf = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=n_jobs
            )
    modelo_rf.fit(X_train, Y_train)
    return modelo_rf

def predicciones(modelo_rf, X_train, X_test):
    Yr_train_RF = modelo_rf.predict(X_train)
    Yr_test_RF = modelo_rf.predict(X_test)

    return Yr_train_RF, Yr_test_RF

def metricas(Y,Yr):
    EG = r2_score(Y, Yr)
    MAE = mean_absolute_error(Y, Yr)
    RMSE = np.sqrt(mean_squared_error(Y, Yr))

    return EG, MAE, RMSE

    