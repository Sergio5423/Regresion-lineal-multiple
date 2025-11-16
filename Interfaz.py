from sklearn.linear_model import LinearRegression
from tkinter import Canvas, ttk
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import regresionLineal as rl
import tkinter as tk
from tkinter import filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import sys
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import re

class ConsoleRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, mensaje):
        self.text_widget.insert(tk.END, mensaje)
        self.text_widget.see(tk.END)  

    def flush(self):
        pass

# Variables globales
dataset_filename = None
inputs = None
outputs = None
w = None
X_train, X_test, Y_train, Y_test = None, None, None, None
Yr_train, Yr_test = None, None
modelo_rl = None
modelo_rf = None

def abrirArchivo():
    global inputs, outputs, dataset_filename

    text_area.delete("1.0", "end")

    filepath = filedialog.askopenfilename(
        title="Seleccionar archivo",
        filetypes=[("CSV files", "*.csv"), ("Todos los archivos", "*.*")],
        initialdir="."
    )

    if not filepath:
        print("No se seleccionó ningún archivo.")
        return

    dataset_filename = os.path.splitext(os.path.basename(filepath))[0]

    try:
        df = pd.read_csv(filepath)
        print("Vista previa del dataset:")
        print(df.head())

        if df.shape[1] < 2:
            print("Error: El dataset debe tener al menos una entrada y una salida.")
            return

        nombre_salida = df.columns[0]
        salida = df[nombre_salida]
        df_entradas = df.drop(columns=[nombre_salida])

        # Codificar cualquier columna de entrada que contenga letras
        for col in df_entradas.columns:
            contiene_letras = df_entradas[col].astype(str).apply(lambda x: bool(re.search(r'[A-Za-z]', x))).any()
            if contiene_letras:
                le = LabelEncoder()
                df_entradas[col] = le.fit_transform(df_entradas[col].astype(str))
                print(f"\nCodificación aplicada a la columna de entrada '{col}':")
                for clase, valor in zip(le.classes_, le.transform(le.classes_)):
                    print(f"   🔸 '{clase}' → {valor}")

        # Codificar la salida si contiene letras
        contiene_letras_salida = salida.astype(str).apply(lambda x: bool(re.search(r'[A-Za-z]', x))).any()
        if contiene_letras_salida:
            le_out = LabelEncoder()
            salida = le_out.fit_transform(salida.astype(str))
            print(f"\nCodificación de la columna de salida '{nombre_salida}':")
            for clase, valor in zip(le_out.classes_, le_out.transform(le_out.classes_)):
                print(f"   🔸 '{clase}' → {valor}")
            
        # Normalizar salida a numpy array
        if isinstance(salida, pd.Series):
            salida = salida.to_numpy()
        else:
            salida = np.array(salida)

        outputs = salida.flatten()
        inputs = df_entradas.values

        lblEntradas.config(text=f"Entradas: {inputs.shape[1]}")
        lblSalidas.config(text="Salidas: 1")
        lblPatrones.config(text=f"Patrones: {inputs.shape[0]}")

        print(f"Dataset cargado: {dataset_filename}")
        
        # Limpiar tabla anterior
        tablaDataset.delete(*tablaDataset.get_children())
        tablaDataset["columns"] = [f"X{i+1}" for i in range(inputs.shape[1])] + ["Yd"]

        for col in tablaDataset["columns"]:
            tablaDataset.heading(col, text=col)
            tablaDataset.column(col, width=80)

        # Insertar filas
        for i in range(len(inputs)):
            fila = list(inputs[i]) + [outputs[i]]
            tablaDataset.insert("", "end", values=fila)        

    except Exception as e:
        print(f"Error al leer el archivo: {e}")

def cargarModelo():
    global Y_test, Yr_test, Y_train, Yr_train, modelo_rl, modelo_rf
    global Yr_train_RF, Yr_test_RF
    global X_train, X_test

    modelo_seleccionado = seleccion_modelo.get()

    try:
        filepath = filedialog.askopenfilename(
            title="Seleccionar modelo entrenado",
            filetypes=[("Model files", "*.pkl"), ("Todos los archivos", "*.*")],
            initialdir="."
        )

        if not filepath:
            textSimulacion.insert("end", "No se seleccionó ningún archivo.\n")
            return
        
        if modelo_seleccionado == "Regresión Lineal Múltiple":
            
            # Cargar el modelo entrenado
            modelo_rl = joblib.load(filepath)
            textSimulacion.insert("end", f"Modelo cargado desde: {filepath}\n")

            # Mostrar coeficientes e intercepto
            if hasattr(modelo_rl, "coef_") and hasattr(modelo_rl, "intercept_"):
                textSimulacion.insert("end", f"Coeficientes: {modelo_rl.coef_}\n")
                textSimulacion.insert("end", f"Intercepto: {modelo_rl.intercept_}\n")

            # Generar predicciones si ya existen conjuntos train/test
            if X_train is not None and Y_train is not None:
                Yr_train = modelo_rl.predict(X_train)
                textSimulacion.insert("end", f"Predicciones de entrenamiento generadas ({len(Yr_train)} patrones).\n")

            if X_test is not None and Y_test is not None:
                Yr_test = modelo_rl.predict(X_test)
                textSimulacion.insert("end", f"Predicciones de prueba generadas ({len(Yr_test)} patrones).\n")
        elif modelo_seleccionado == "Random Forest Regressor":
            
            # Cargar el modelo entrenado
            modelo_rf = joblib.load(filepath)
            textSimulacionRF.insert("end", f"Modelo cargado desde: {filepath}\n")

            # Generar predicciones si ya existen conjuntos train/test
            if X_train is not None and Y_train is not None:
                Yr_train_RF = modelo_rf.predict(X_train)
                textSimulacionRF.insert("end", f"Predicciones de entrenamiento generadas ({len(Yr_train_RF)} patrones).\n")

            if X_test is not None and Y_test is not None:
                Yr_test_RF = modelo_rf.predict(X_test)  
                textSimulacionRF.insert("end", f"Predicciones de prueba generadas ({len(Yr_test_RF)} patrones).\n")

    except Exception as e:
        textSimulacion.insert("end", f"Error al cargar el modelo: {e}\n")

    except Exception as e:
        textSimulacion.insert("end", f"Error al cargar el modelo: {e}\n")

    except Exception as e:
        textSimulacion.insert("end", f"Error al cargar el modelo: {e}\n")


def almacenarModelo(modelo, carpeta_destino="Modelos"):
    global dataset_filename

    modelo_seleccionado = seleccion_modelo.get()

    if not messagebox.askyesno("Guardar Modelo", "¿Desea guardar el modelo entrenado?"):
        text_entrenamiento.insert("end", f"Guardado cancelado.\n")
        return

    os.makedirs(carpeta_destino, exist_ok=True)

    base_name = dataset_filename if dataset_filename else "sin_nombre"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model = modelo_seleccionado.replace(" ", "_")
    nombre_archivo = f"{base_name}_{timestamp}_{model}.pkl"   # extensión correcta
    ruta_archivo = os.path.join(carpeta_destino, nombre_archivo)

    # Guardar el modelo entrenado con joblib
    joblib.dump(modelo, ruta_archivo)

    text_entrenamiento.insert("end", f"Modelo guardado en: {ruta_archivo}\n")

def entrenar():
    global Y_test, Yr_test, Y_train, Yr_train, modelo_rl, modelo_rf
    global Yr_train_RF, Yr_test_RF

    modelo_seleccionado = seleccion_modelo.get()

    if modelo_seleccionado == "Regresión Lineal Múltiple":

        try:
            text_entrenamiento.insert("end", "Entrenando Regresión Lineal Múltiple...\n")

            # Entrenar modelo directamente
            modelo_rl = LinearRegression()
            modelo_rl.fit(X_train, Y_train)

            # Predicciones
            Yr_train = modelo_rl.predict(X_train)
            Yr_test = modelo_rl.predict(X_test)

            # Métricas
            EG_train = r2_score(Y_train, Yr_train)
            EG_test = r2_score(Y_test, Yr_test)

            MAE_train = mean_absolute_error(Y_train, Yr_train)
            MAE_test = mean_absolute_error(Y_test, Yr_test)

            RMSE_train = np.sqrt(mean_squared_error(Y_train, Yr_train))
            RMSE_test = np.sqrt(mean_squared_error(Y_test, Yr_test))

            # Mostrar resultados
            lblError.config(text=f"EG Entrenamiento: {EG_train:.4f} \nEG Prueba: {EG_test:.4f}", foreground="blue")
            lblMAE.config(text=f"MAE Entrenamiento: {MAE_train:.4f} \nMAE Prueba: {MAE_test:.4f}", foreground="darkgreen")
            lblRMSE.config(text=f"RMSE Entrenamiento: {RMSE_train:.4f} \nRMSE Prueba: {RMSE_test:.4f}", foreground="purple")

            text_entrenamiento.insert("end", "Entrenamiento completado.\n")
            text_entrenamiento.insert("end", f"Coeficientes: {modelo_rl.coef_}\n")
            text_entrenamiento.insert("end", f"Intercepto: {modelo_rl.intercept_}\n")

            # Graficar
            graficar_hist_residuales(Y_train, Yr_train)
            graficar_residuales(Y_train, Yr_train)
            graficar_dispersion(Y_train, Yr_train)

            almacenarModelo(modelo_rl)

        except Exception as e:
            text_entrenamiento.insert("end", f"Error durante el entrenamiento RL: {e}\n")

    elif modelo_seleccionado == "Random Forest Regressor":
        try:
            text_entrenamiento_RF.insert("end", "Entrenando Random Forest Regressor...\n")

            modelo_rf = RandomForestRegressor(
                n_estimators=200,
                max_depth=12,
                random_state=42,
                n_jobs=-1
            )
            modelo_rf.fit(X_train, Y_train)

            Yr_train_RF = modelo_rf.predict(X_train)
            Yr_test_RF = modelo_rf.predict(X_test)

            EG_train = r2_score(Y_train, Yr_train_RF)
            EG_test = r2_score(Y_test, Yr_test_RF)
            MAE_train = mean_absolute_error(Y_train, Yr_train_RF)
            MAE_test = mean_absolute_error(Y_test, Yr_test_RF)
            RMSE_train = np.sqrt(mean_squared_error(Y_train, Yr_train_RF))
            RMSE_test = np.sqrt(mean_squared_error(Y_test, Yr_test_RF))

            lblErrorRF.config(text=f"EG Entrenamiento: {EG_train:.4f} \nEG Prueba: {EG_test:.4f}", foreground="blue")
            lblMAERF.config(text=f"MAE Entrenamiento: {MAE_train:.4f} \nMAE Prueba: {MAE_test:.4f}", foreground="darkgreen")
            lblRMSERF.config(text=f"RMSE Entrenamiento: {RMSE_train:.4f} \nRMSE Prueba: {RMSE_test:.4f}", foreground="purple")

            text_entrenamiento_RF.insert("end", "Entrenamiento completado.\n")            

            graficar_hist_residuales_RF(Y_train, Yr_train_RF)
            graficar_residuales_RF(Y_train, Yr_train_RF)
            graficar_dispersion_RF(Y_train, Yr_train_RF)

            almacenarModelo(modelo_rf)

        except Exception as e:
            text_entrenamiento_RF.insert("end", f"Error durante el entrenamiento RF: {e}\n")

def calcular_metricas(Yd, Yr):
    errores = [abs(y1 - y2) for y1, y2 in zip(Yd, Yr)]
    EG = sum(errores) / len(errores)
    MAE = sum(abs(y1 - y2) for y1, y2 in zip(Yd, Yr)) / len(Yd)
    RMSE = (sum((y1 - y2)**2 for y1, y2 in zip(Yd, Yr)) / len(Yd))**0.5
    return EG, MAE, RMSE

def procesar_entrada(texto):
    codigos_barrios = {
        "La Nevada": 0,
        "Los Cortijos": 1,
        "Los Fundadores": 2,
        "Novalito": 3,
        "Rosario": 4,
        "Villa Miriam": 5
    }

    valores = [x.strip() for x in texto.split(",")]
    if len(valores) != 10:
        raise ValueError("Número de entradas incorrecto. Se esperaban 10 valores.")

    area = float(valores[0])
    habitaciones = float(valores[1])
    banos = float(valores[2])
    antiguedad = float(valores[3])
    estrato = float(valores[4])
    parqueadero = float(valores[5])
    ascensor = float(valores[6])
    barrio = codigos_barrios[valores[7]]
    cercania = float(valores[8])
    tasa_desempleo = float(valores[9])

    entrada = [area, habitaciones, banos, antiguedad, estrato,
               parqueadero, ascensor, barrio, cercania, tasa_desempleo]

    return entrada, valores[7]  # devuelve también el nombre del barrio

def simular():
    global modelo_rl, modelo_rf

    modelo_seleccionado = seleccion_modelo.get()
    texto = inpSimulacion.get().strip()

    if not texto:
        textSimulacion.insert("end", "Por favor ingresa valores separados por coma.\n")
        return

    try:
        entrada, barrio_nombre = procesar_entrada(texto)
        entrada_array = np.array(entrada).reshape(1, -1)

        if modelo_seleccionado == "Regresión Lineal Múltiple":
            if modelo_rl is None:
                textSimulacion.insert("end", "No hay modelo RL cargado o entrenado.\n")
                return
            resultado = modelo_rl.predict(entrada_array)[0]
            textSimulacion.insert("end", f"Entrada: {entrada}\nBarrio: {barrio_nombre}\nPredicción (RL): {resultado:.2f}\n\n")

        elif modelo_seleccionado == "Random Forest Regressor":
            if modelo_rf is None:
                textSimulacion.insert("end", "No hay modelo RF cargado o entrenado.\n")
                return
            resultado = modelo_rf.predict(entrada_array)[0]
            textSimulacion.insert("end", f"Entrada: {entrada}\nBarrio: {barrio_nombre}\nPredicción (RF): {resultado:.2f}\n\n")

    except ValueError as ve:
        textSimulacion.insert("end", f"Error: {ve}\n\n")
    except Exception as e:
        textSimulacion.insert("end", f"Error durante la simulación: {e}\n\n")

def simular_prueba():
    global Yr_test, Y_test, X_test
    global modelo_rf, modelo_rl
    global axS1, axS2, axS3, canvasS1, canvasS2, canvasS3

    modelo_seleccionado = seleccion_modelo.get()

    # Verificar si tenemos datos de prueba
    if X_test is None or Y_test is None:
        textSimulacion.insert("end", "No hay conjunto de prueba disponible. Primero divida el dataset.\n")
        textSimulacionRF.insert("end", "No hay conjunto de prueba disponible. Primero divida el dataset.\n")
        return

    # Si no tenemos Yr_test pero tenemos modelo y X_test, calcularlo
    if Yr_test is None and modelo_rl is not None and X_test is not None:
        Yr_test = modelo_rl.predict(X_test)
        textSimulacion.insert("end", "Predicciones de prueba REGRSION LINEAL MÚLTIPLE calculadas a partir del modelo cargado.\n")
    elif Yr_test is None and modelo_rf is not None and X_test is not None:
        Yr_test = modelo_rf.predict(X_test)
        textSimulacionRF.insert("end", "Predicciones de prueba RANDOM FOREST REGRESSOR calculadas a partir del modelo cargado.\n")
    
    # Si aún no tenemos Yr_test, mostrar error
    if Yr_test is None:
        textSimulacion.insert("end", "No hay resultados de prueba disponibles. Entrene un modelo o cargue uno con predicciones.\n")
        textSimulacionRF.insert("end", "No hay resultados de prueba disponibles. Entrene un modelo o cargue uno con predicciones.\n")
        return

    # Mostrar resultados en consola
    if modelo_seleccionado == "Regresión Lineal Múltiple":
        textSimulacion.delete("1.0", "end")  # limpiar consola
        for i in range(len(Y_test)):
            textSimulacion.insert("end", f"Patrón {i+1} | Yd: {Y_test[i]:.2f} | Yr: {Yr_test[i]:.2f}\n")
    elif modelo_seleccionado == "Random Forest Regressor":
        textSimulacionRF.delete("1.0", "end")  # limpiar consola
        for i in range(len(Y_test)):
            textSimulacionRF.insert("end", f"Patrón {i+1} | Yd: {Y_test[i]:.2f} | Yr: {Yr_test_RF[i]:.2f}\n")

    if modelo_seleccionado == "Regresión Lineal Múltiple":    
        # Calcular métricas globales
        mae = mean_absolute_error(Y_test, Yr_test)
        rmse = np.sqrt(mean_squared_error(Y_test, Yr_test))
        r2 = r2_score(Y_test, Yr_test)

        # Mostrar error de simulación
        lblErrorSim.config(text=(
            f"EG Simulación: {r2:.4f}\n"        
        ), foreground="blue")

        lblMaeSim.config(text=(
            f"MAE Simulación: {mae:.4f}\n"        
        ), foreground="darkgreen")

        lblRmseSim.config(text=(
            f"RMSE Simulación: {rmse:.4f} \n"        
        ), foreground="purple")

        # Graficar resultados
        actualizar_graficas_simulacion(modelo_seleccionado, np.array(Y_test), np.array(Yr_test))

    elif modelo_seleccionado == "Random Forest Regressor":           
        # Calcular métricas globales
        maeRF = mean_absolute_error(Y_test, Yr_test_RF)
        rmseRF = np.sqrt(mean_squared_error(Y_test, Yr_test_RF))
        r2RF = r2_score(Y_test, Yr_test_RF)

        # Mostrar error de simulación
        lblErrorSimRF.config(text=(
            f"EG Simulación: {r2RF:.4f}\n"        
        ), foreground="blue")

        lblMaeSimRF.config(text=(
            f"MAE Simulación: {maeRF:.4f}\n"        
        ), foreground="darkgreen")

        lblRmseSimRF.config(text=(
            f"RMSE Simulación: {rmseRF:.4f} \n"        
        ), foreground="purple")

        # Graficar resultados
        actualizar_graficas_simulacion(modelo_seleccionado, np.array(Y_test), np.array(Yr_test_RF))        

def actualizar_graficas_simulacion(modelo_seleccionado, Yd, Yr):
    residuales = Yd - Yr

    if modelo_seleccionado == "Regresión Lineal Múltiple":
        
        # Histograma
        axS1.clear()
        axS1.set_title("Distribución de Residuales (Simulación)")
        axS1.set_xlabel("Error (Yd - Yr)")
        axS1.set_ylabel("Frecuencia")
        axS1.hist(residuales, bins=20, color="skyblue", edgecolor="black")
        canvasS1.draw()

        # Residuales por patrón
        axS2.clear()
        axS2.set_title("Residuales por Patrón (Simulación)")
        axS2.set_xlabel("Índice de patrón")
        axS2.set_ylabel("Error (Yd - Yr)")
        axS2.scatter(range(len(residuales)), residuales, color="purple", alpha=0.6)
        axS2.axhline(y=0, color="gray", linestyle="--")
        canvasS2.draw()

        # Dispersión Yd vs Yr
        axS3.clear()
        axS3.set_title("Dispersión Yd vs Yr (Simulación)")
        axS3.set_xlabel("Salida Deseada (Yd)")
        axS3.set_ylabel("Salida Obtenida (Yr)")
        axS3.scatter(Yd, Yr, color="green", alpha=0.6)
        axS3.plot([min(Yd), max(Yd)], [min(Yd), max(Yd)], color="red", linestyle="--")
        canvasS3.draw()

    elif modelo_seleccionado == "Random Forest Regressor":
        # Histograma
        axSRF1.clear()
        axSRF1.set_title("Distribución de Residuales (Simulación)")
        axSRF1.set_xlabel("Error (Yd - Yr)")
        axSRF1.set_ylabel("Frecuencia")
        axSRF1.hist(residuales, bins=20, color="skyblue", edgecolor="black")
        canvasSRF1.draw()

        # Residuales por patrón
        axSRF2.clear()
        axSRF2.set_title("Residuales por Patrón (Simulación)")
        axSRF2.set_xlabel("Índice de patrón")
        axSRF2.set_ylabel("Error (Yd - Yr)")
        axSRF2.scatter(range(len(residuales)), residuales, color="purple", alpha=0.6)
        axSRF2.axhline(y=0, color="gray", linestyle="--")
        canvasSRF2.draw()

        # Dispersión Yd vs Yr
        axSRF3.clear()
        axSRF3.set_title("Dispersión Yd vs Yr (Simulación)")
        axSRF3.set_xlabel("Salida Deseada (Yd)")
        axSRF3.set_ylabel("Salida Obtenida (Yr)")
        axSRF3.scatter(Yd, Yr, color="green", alpha=0.6)
        axSRF3.plot([min(Yd), max(Yd)], [min(Yd), max(Yd)], color="red", linestyle="--")
        canvasSRF3.draw()

def reiniciarModelo():
    global modelo, Y_train, Yr_train, Y_test, Yr_test
    global ax1, ax2, ax3, canvas1, canvas2, canvas3
    global axRF1, axRF2, axRF3, canvasRF1, canvasRF2, canvasRF3
    global axS1, axS2, axS3, canvasS1, canvasS2, canvasS3

    # Reiniciar modelo y datos
    modelo = None
    Y_train, Yr_train, Y_test, Yr_test = None, None, None, None

    # Limpiar consola principal
    text_area.delete("1.0", "end")
    text_area.insert("end", "Modelo reiniciado.\n")

    # Limpiar consola de simulación
    textSimulacion.delete("1.0", "end")
    textSimulacion.insert("end", "Simulación reiniciada.\n")

    # Limpiar gráficas de entrenamiento
    ax1.clear()
    ax1.set_title("Distribución de Residuales")
    ax1.set_xlabel("Error (Real - Predicho)")
    ax1.set_ylabel("Frecuencia")
    canvas1.draw()

    ax2.clear()
    ax2.set_title("Gráfica de Residuales")
    ax2.set_xlabel("Índice de patrón")
    ax2.set_ylabel("Error (Real - Predicho)")
    canvas2.draw()

    ax3.clear()
    ax3.set_title("Dispersión de Predicciones")
    ax3.set_xlabel("Salida Deseada (Yd)")
    ax3.set_ylabel("Salida Obtenida (Yr)")
    canvas3.draw()

    axRF1.clear()
    axRF1.set_title("Distribución de Residuales")
    axRF1.set_xlabel("Error (Real - Predicho)")
    axRF1.set_ylabel("Frecuencia")
    canvasRF1.draw()

    axRF2.clear()
    axRF2.set_title("Gráfica de Residuales")
    axRF2.set_xlabel("Índice de patrón")
    axRF2.set_ylabel("Error (Real - Predicho)")
    canvasRF2.draw()

    axRF3.clear()
    axRF3.set_title("Dispersión de Predicciones")
    axRF3.set_xlabel("Salida Deseada (Yd)")
    axRF3.set_ylabel("Salida Obtenida (Yr)")
    canvasRF3.draw()

    # Limpiar gráficas de simulación
    axS1.clear()
    axS1.set_title("Distribución de Residuales (Simulación)")
    axS1.set_xlabel("Error (Yd - Yr)")
    axS1.set_ylabel("Frecuencia")
    canvasS1.draw()

    axS2.clear()
    axS2.set_title("Residuales por Patrón (Simulación)")
    axS2.set_xlabel("Índice de patrón")
    axS2.set_ylabel("Error (Yd - Yr)")
    canvasS2.draw()

    axS3.clear()
    axS3.set_title("Dispersión Yd vs Yr (Simulación)")
    axS3.set_xlabel("Salida Deseada (Yd)")
    axS3.set_ylabel("Salida Obtenida (Yr)")
    canvasS3.draw()

    lblError.config(text=(
        f"EG Entrenamiento: \n"
        f"EG Prueba: "
    ), foreground="blue")

    lblMAE.config(text=(
        f"MAE Entrenamiento:  \n"
        f"MAE Prueba: "
    ), foreground="darkgreen")

    lblRMSE.config(text=(
        f"RMSE Entrenamiento: \n"
        f"RMSE Prueba: "
    ), foreground="purple")    

    lblErrorRF.config(text=(
        f"EG Entrenamiento: \n"
        f"EG Prueba: "
    ), foreground="blue")

    lblMAERF.config(text=(
        f"MAE Entrenamiento:  \n"
        f"MAE Prueba: "
    ), foreground="darkgreen")

    lblRMSERF.config(text=(
        f"RMSE Entrenamiento: \n"
        f"RMSE Prueba: "
    ), foreground="purple")    

    dividir_dataset(inputs, outputs)

    print("Modelo y gráficas reiniciados correctamente.")


def graficar_errores(EG, inpError):
    ax2.clear()
    ax2.set_title("Error General vs Error Óptimo")
    ax2.set_xlabel("Tipo de Error")
    ax2.set_ylabel("Valor")

    tipos = ["Error General", "Error Óptimo"]
    valores = [EG, inpError]

    ax2.bar(tipos, valores, color=["orange", "green"])
    for i, valor in enumerate(valores):
        ax2.text(i, valor + 0.01, f"{valor:.4f}", ha="center", va="bottom", fontsize=10)

    canvas2.draw()

def graficar_yd_vs_yr(Yd, Yr):
    ax1.clear()
    ax1.set_title("Yd vs Yr por patrón")
    ax1.set_xlabel("Patrón")
    ax1.set_ylabel("Salida")

    patrones = list(range(1, len(Yd) + 1))

    ax1.plot(patrones, Yd, label="Yd (Deseada)", marker="o", color="blue")
    ax1.plot(patrones, Yr, label="Yr (Obtenida)", marker="x", color="red")

    ax1.legend()
    canvas1.draw()

def graficar_hist_residuales(Y_real, Y_pred):
    residuales = Y_real - Y_pred
    ax1.clear()
    ax1.set_title("Distribución de Residuales")
    ax1.set_xlabel("Error (Real - Predicho)")
    ax1.set_ylabel("Frecuencia")
    ax1.hist(residuales, bins=20, color="skyblue", edgecolor="black", alpha=0.7)
    canvas1.draw()

def graficar_residuales(Y_real, Y_pred):
    residuales = Y_real - Y_pred
    ax2.clear()
    ax2.set_title("Gráfica de Residuales")
    ax2.set_xlabel("Índice de patrón")
    ax2.set_ylabel("Error (Real - Predicho)")
    ax2.scatter(range(len(residuales)), residuales, color="blue", alpha=0.6)
    ax2.axhline(y=0, color="red", linestyle="--")
    canvas2.draw()

def graficar_dispersion(Y_real, Y_pred):
    ax3.clear()
    ax3.set_title("Dispersión de Predicciones")
    ax3.set_xlabel("Salida Deseada (Yd)")
    ax3.set_ylabel("Salida Obtenida (Yr)")
    ax3.scatter(Y_real, Y_pred, color="green", alpha=0.6)
    ax3.plot([Y_real.min(), Y_real.max()], [Y_real.min(), Y_real.max()], color="red", linestyle="--")
    canvas3.draw()

def graficar_hist_residuales_RF(Y_real, Y_pred):
    residuales = Y_real - Y_pred
    axRF1.clear()
    axRF1.set_title("Distribución de Residuales")
    axRF1.set_xlabel("Error (Real - Predicho)")
    axRF1.set_ylabel("Frecuencia")
    axRF1.hist(residuales, bins=20, color="skyblue", edgecolor="black", alpha=0.7)
    canvasRF1.draw()

def graficar_residuales_RF(Y_real, Y_pred):
    residuales = Y_real - Y_pred
    axRF2.clear()
    axRF2.set_title("Gráfica de Residuales")
    axRF2.set_xlabel("Índice de patrón")
    axRF2.set_ylabel("Error (Real - Predicho)")
    axRF2.scatter(range(len(residuales)), residuales, color="blue", alpha=0.6)
    axRF2.axhline(y=0, color="red", linestyle="--")
    canvasRF2.draw()

def graficar_dispersion_RF(Y_real, Y_pred):
    axRF3.clear()
    axRF3.set_title("Dispersión de Predicciones")
    axRF3.set_xlabel("Salida Deseada (Yd)")
    axRF3.set_ylabel("Salida Obtenida (Yr)")
    axRF3.scatter(Y_real, Y_pred, color="green", alpha=0.6)
    axRF3.plot([Y_real.min(), Y_real.max()], [Y_real.min(), Y_real.max()], color="red", linestyle="--")
    canvasRF3.draw()


def dividir_dataset(X, Y):
    global X_train, X_test, Y_train, Y_test

    porcentaje_str = seleccion.get()
    porcentaje_entrenamiento = int(porcentaje_str.strip('%')) / 100.0
    print(f"Porcentaje de entrenamiento: {porcentaje_entrenamiento}")
    
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, train_size=porcentaje_entrenamiento, random_state=42
    )
    print(f"División del dataset:")
    print(f"   - Entrenamiento: {len(X_train)} patrones")    
    print(f"   - Prueba: {len(X_test)} patrones")    

def crear_ventana():
    global ax1, ax2, ax3, canvas1, canvas2, canvas3, text_area, tablaDataset, text_entrenamiento
    global axRF1, axRF2, axRF3, canvasRF1, canvasRF2, canvasRF3, text_entrenamiento_RF
    global lblEntradas, lblSalidas, lblPatrones
    global lblError, lblMAE, lblRMSE
    global lblErrorSim, lblMaeSim, lblRmseSim
    global lblErrorRF, lblMAERF, lblRMSERF
    global lblErrorSimRF, lblMaeSimRF, lblRmseSimRF
    global inpSimulacion, textSimulacion, seleccion_modelo, seleccion
    global axS1, axS2, axS3, canvasS1, canvasS2, canvasS3
    global axSRF1, axSRF2, axSRF3, canvasSRF1, canvasSRF2, canvasSRF3, textSimulacionRF

    root = tk.Tk()
    root.title("Regresión Lineal Múltiple")
    root.geometry("1323x1004")

    # Crear notebook (pestañas)
    notebook = ttk.Notebook(root)
    notebook.pack(fill='both', expand=True)

    # Pestañas
    frame_controles = ttk.Frame(notebook)
    frame_entrenamiento = ttk.Frame(notebook)    
    frame_simulacion = ttk.Frame(notebook)
    frame_dashboard = ttk.Frame(notebook)

    notebook.add(frame_controles, text="Controles")
    notebook.add(frame_entrenamiento, text="Entrenamiento y Evaluación")    
    notebook.add(frame_simulacion, text="Simulación")
    notebook.add(frame_dashboard, text="Dashboard")

    # ---------------------- BOTONES SUPERIORES ----------------------
    frame_botones = ttk.LabelFrame(frame_controles, text="Acciones")
    frame_botones.grid(column=0, row=0, columnspan=3, padx=10, pady=10, sticky="ew")

    btnCargarDS = ttk.Button(frame_botones, text="Cargar Dataset", command=abrirArchivo)
    btnCargarDS.grid(column=0, row=0, padx=5, pady=5)

    btnCargarW = ttk.Button(frame_botones, text="Cargar Modelo", command=cargarModelo)
    btnCargarW.grid(column=1, row=0, padx=5, pady=5)

    opciones = ["60%", "70%", "80%", "90%"]
    seleccion = tk.StringVar()
    ttk.Label(frame_botones, text="Porcentaje de entrenamiento:").grid(column=2, row=0, padx=5, pady=5)
    dbPorcentaje = ttk.Combobox(frame_botones, textvariable=seleccion, values=opciones, state="readonly")
    dbPorcentaje.grid(column=3, row=0, padx=5, pady=5)
    dbPorcentaje.current(1)

    btnDividir = ttk.Button(frame_botones, text="Dividir Dataset", command=lambda: dividir_dataset(inputs, outputs))
    btnDividir.grid(column=4, row=0, padx=5, pady=5)

    # ---------------------- INFORMACIÓN DEL DATASET ----------------------
    frame_info = ttk.LabelFrame(frame_controles, text="Información del Dataset")
    frame_info.grid(column=0, row=1, columnspan=2, padx=10, pady=10, sticky="nw")

    lblEntradas = ttk.Label(frame_info, text="Entradas: 0")
    lblEntradas.grid(column=0, row=0, padx=10, pady=5, sticky="w")

    lblSalidas = ttk.Label(frame_info, text="Salidas: 0")
    lblSalidas.grid(column=0, row=1, padx=10, pady=5, sticky="w")

    lblPatrones = ttk.Label(frame_info, text="Patrones: 0")
    lblPatrones.grid(column=0, row=2, padx=10, pady=5, sticky="w")

    # Configurar expansión en frame_controles
    frame_controles.columnconfigure(2, weight=1)
    frame_controles.rowconfigure(1, weight=1)

    # ---------------------- CONSOLA ----------------------
    text_area = tk.Text(frame_controles, wrap="word")
    text_area.grid(column=2, row=1, rowspan=3, padx=10, pady=10, sticky="nsew")
    sys.stdout = ConsoleRedirector(text_area)

    # ---------------------- TABLA DEL DATASET ----------------------
    frame_dataset = ttk.LabelFrame(frame_controles, text="Vista del Dataset")
    frame_dataset.grid(column=2, row=5, padx=10, pady=10, sticky="nsew")

    tablaDataset = ttk.Treeview(frame_dataset, show="headings", height=10)
    tablaDataset.pack(fill="both", expand=True)

    # ---------------------- ENTRENAMIENTO Y EVALUACIÓN ----------------------
    frame_ent_btn = ttk.LabelFrame(frame_entrenamiento, text="Parámetros del entrenamiento")
    frame_ent_btn.grid(column=0, row=0, columnspan=3, padx=10, pady=10, sticky="ew")

    opModelos = ["Regresión Lineal Múltiple", "Random Forest Regressor"]
    seleccion_modelo = tk.StringVar()
    ttk.Label(frame_ent_btn, text="Modelo:").grid(column=0, row=0, padx=5, pady=5)
    dbModelos = ttk.Combobox(frame_ent_btn, textvariable=seleccion_modelo, values=opModelos, state="readonly")
    dbModelos.grid(column=1, row=0, padx=5, pady=5)
    dbModelos.current(0)

    btnReiniciar = ttk.Button(frame_ent_btn, text="Reiniciar Modelo", command=reiniciarModelo)
    btnReiniciar.grid(column=2, row=0, padx=5, pady=5)

    """
    ttk.Label(frame_ent_btn, text="Error Máximo:").grid(column=3, row=0, padx=10, pady=5)
    inpError = ttk.Entry(frame_ent_btn)
    inpError.grid(column=4, row=0, padx=10, pady=5)
    """

    btnEntrenar = ttk.Button(frame_ent_btn, text="Entrenar", command=entrenar)
    btnEntrenar.grid(column=5, row=0, padx=5, pady=5)

    # ---------------------- MÉTRICAS DE REGRESION LINEAL MÚLTIPLE----------------------
    frame_met = ttk.LabelFrame(frame_entrenamiento, text="Métricas de Entrenamiento Regresión Lineal Múltiple")
    frame_met.grid(column=0, row=1, padx=10, pady=10, sticky="nw")
    frame_entrenamiento.columnconfigure(0, minsize=250)

    lblError = ttk.Label(frame_met, text="Error", foreground="blue")
    lblError.grid(column=0, row=0, padx=10, pady=5, sticky="w")

    lblMAE = ttk.Label(frame_met, text="MAE", foreground="blue")
    lblMAE.grid(column=0, row=1, padx=10, pady=5, sticky="w")

    lblRMSE = ttk.Label(frame_met, text="RMSE", foreground="blue")
    lblRMSE.grid(column=0, row=2, padx=10, pady=5, sticky="w") 

    #---------------------- MÉTRICAS DE RANDOM FOREST REGRESSOR ----------------------
    frame_met_rf = ttk.LabelFrame(frame_entrenamiento, text="Métricas de Random Forest Regressor")
    frame_met_rf.grid(column=0, row=1, padx=10, pady=200, sticky="nw")

    lblErrorRF = ttk.Label(frame_met_rf, text="Error", foreground="blue")
    lblErrorRF.grid(column=0, row=0, padx=10, pady=5, sticky="w")

    lblMAERF = ttk.Label(frame_met_rf, text="MAE", foreground="blue")
    lblMAERF.grid(column=0, row=1, padx=10, pady=5, sticky="w")

    lblRMSERF = ttk.Label(frame_met_rf, text="RMSE", foreground="blue")
    lblRMSERF.grid(column=0, row=2, padx=10, pady=5, sticky="w")

    # ---------------------- PESTAÑAS DE GRÁFICAS ENTRENAMIENTO----------------------

    notebook_graficas = ttk.Notebook(frame_entrenamiento)
    notebook_graficas.grid(column=1, row=1, padx=10, pady=10, sticky="nsew")   

    frame_regresion = ttk.Frame(notebook_graficas)
    frame_random = ttk.Frame(notebook_graficas)

    notebook_graficas.add(frame_regresion, text="Regresión Lineal Múltiple")
    notebook_graficas.add(frame_random, text="Random Forest Regressor")

    # ---------------------- GRÁFICAS ENTRENAMIENTO REGRESION LINEAL----------------------
    fig1, ax1 = plt.subplots(figsize=(5, 4))
    fig2, ax2 = plt.subplots(figsize=(5, 4))
    fig3, ax3 = plt.subplots(figsize=(5, 4))

    ax1.set_title("Distribución de Residuales")
    ax1.set_xlabel("Error (Real - Predicho)")
    ax1.set_ylabel("Frecuencia")

    # Inicializar gráfica de residuales vacía
    ax2.set_title("Gráfica de Residuales - Esperando datos")
    ax2.set_xlabel("Índice de patrón")
    ax2.set_ylabel("Error (Real - Predicho)")
    ax2.text(0.5, 0.5, "Ejecuta el entrenamiento\npara ver los residuales",
            ha='center', va='center', transform=ax2.transAxes, fontsize=12)
    fig2.tight_layout()

    ax3.set_title("Dispersión de Predicciones (Entrenamiento)")
    ax3.set_xlabel("Salida Deseada (Yd)")
    ax3.set_ylabel("Salida Obtenida (Yr)")
    fig3.tight_layout()

    canvas1 = FigureCanvasTkAgg(fig1, master=frame_regresion)
    canvas2 = FigureCanvasTkAgg(fig2, master=frame_regresion)
    canvas3 = FigureCanvasTkAgg(fig3, master=frame_regresion)

    canvas1.get_tk_widget().grid(row=0, column=0, padx=10, pady=10)
    canvas2.get_tk_widget().grid(row=0, column=1, padx=10, pady=10)
    canvas3.get_tk_widget().grid(row=1, column=0, padx=10, pady=10)

    text_entrenamiento = tk.Text(frame_regresion, wrap="word", width=60)
    text_entrenamiento.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")    

    # ---------------------- GRÁFICAS ENTRENAMIENTO RANDOM FOREST----------------------
    figRF1, axRF1 = plt.subplots(figsize=(5, 4))
    figRF2, axRF2 = plt.subplots(figsize=(5, 4))
    figRF3, axRF3 = plt.subplots(figsize=(5, 4))

    axRF1.set_title("Distribución de Residuales")
    axRF1.set_xlabel("Error (Real - Predicho)")
    axRF1.set_ylabel("Frecuencia")

    # Inicializar gráfica de residuales vacía
    axRF2.set_title("Gráfica de Residuales - Esperando datos")
    axRF2.set_xlabel("Índice de patrón")
    axRF2.set_ylabel("Error (Real - Predicho)")
    axRF2.text(0.5, 0.5, "Ejecuta el entrenamiento\npara ver los residuales",
            ha='center', va='center', transform=axRF2.transAxes, fontsize=12)
    figRF2.tight_layout()

    axRF3.set_title("Dispersión de Predicciones (Entrenamiento)")
    axRF3.set_xlabel("Salida Deseada (Yd)")
    axRF3.set_ylabel("Salida Obtenida (Yr)")
    figRF3.tight_layout()

    canvasRF1 = FigureCanvasTkAgg(figRF1, master=frame_random)
    canvasRF2 = FigureCanvasTkAgg(figRF2, master=frame_random)
    canvasRF3 = FigureCanvasTkAgg(figRF3, master=frame_random)

    canvasRF1.get_tk_widget().grid(row=0, column=0, padx=10, pady=10)
    canvasRF2.get_tk_widget().grid(row=0, column=1, padx=10, pady=10)
    canvasRF3.get_tk_widget().grid(row=1, column=0, padx=10, pady=10)

    #------------------------ CONSOLA --------------------------
    text_entrenamiento_RF = tk.Text(frame_random, wrap="word", width=60)
    text_entrenamiento_RF.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")    

    # ---------------------- SIMULACIÓN ----------------------
    frame_sim_input = ttk.LabelFrame(frame_simulacion, text="Entrada del Patrón")
    frame_sim_input.grid(column=0, row=0, padx=10, pady=10, sticky="ew", columnspan=2)

    ttk.Label(frame_sim_input, text="Modelo:").grid(column=0, row=0, padx=5, pady=5)
    dbModelos = ttk.Combobox(frame_sim_input, textvariable=seleccion_modelo, values=opModelos, state="readonly")
    dbModelos.grid(column=1, row=0, padx=5, pady=5)
    dbModelos.current(0)

    ttk.Label(frame_sim_input, text="Entradas (separadas por coma):").grid(column=2, row=0, padx=10, pady=10)
    inpSimulacion = ttk.Entry(frame_sim_input, width=50)
    inpSimulacion.grid(column=3, row=0, padx=5, pady=5)

    btnSimular = ttk.Button(frame_sim_input, text="Simular", command=simular)
    btnSimular.grid(column=4, row=0, padx=5, pady=5)

    btnSimularPrueba = ttk.Button(frame_sim_input, text="Simular Conjunto de Prueba", command=simular_prueba)
    btnSimularPrueba.grid(column=5, row=0, padx=5, pady=5)

    btnCargarWU = ttk.Button(frame_sim_input, text="Cargar Modelo", command=cargarModelo)
    btnCargarWU.grid(column=6, row=0, padx=5, pady=5)

    # ---------------------- MÉTRICAS SIMULACIÓN REGRESIÓN LINEAL----------------------
    frame_met_sim = ttk.LabelFrame(frame_simulacion, text="Métricas de Simulación regresión lineal múltiple")
    frame_met_sim.grid(column=0, row=1, padx=10, pady=10, sticky="nw")
    frame_simulacion.columnconfigure(0, minsize=250)

    lblErrorSim = ttk.Label(frame_met_sim, text="Error", foreground="blue")
    lblErrorSim.grid(column=0, row=0, padx=10, pady=5, sticky="w")

    lblMaeSim = ttk.Label(frame_met_sim, text="MAE", foreground="blue")
    lblMaeSim.grid(column=0, row=1, padx=10, pady=5, sticky="w")

    lblRmseSim = ttk.Label(frame_met_sim, text="RMSE", foreground="blue")
    lblRmseSim.grid(column=0, row=2, padx=10, pady=5, sticky="w")

    # ---------------------- MÉTRICAS SIMULACIÓN RANDOM FOREST REGRESSOR----------------------
    frame_met_sim_rf = ttk.LabelFrame(frame_simulacion, text="Métricas de Simulación random forest regressor")
    frame_met_sim_rf.grid(column=0, row=1, padx=10, pady=200, sticky="nw")

    lblErrorSimRF = ttk.Label(frame_met_sim_rf, text="Error", foreground="blue")
    lblErrorSimRF.grid(column=0, row=0, padx=10, pady=5, sticky="w")

    lblMaeSimRF = ttk.Label(frame_met_sim_rf, text="MAE", foreground="blue")
    lblMaeSimRF.grid(column=0, row=1, padx=10, pady=5, sticky="w")

    lblRmseSimRF = ttk.Label(frame_met_sim_rf, text="RMSE", foreground="blue")
    lblRmseSimRF.grid(column=0, row=2, padx=10, pady=5, sticky="w")

    # ---------------------- PESTAÑAS DE GRÁFICAS SIMULACIÓN ----------------------

    notebook_graficas_sim = ttk.Notebook(frame_simulacion)
    notebook_graficas_sim.grid(column=1, row=1, padx=10, pady=10, sticky="nsew")
    
    frame_regresion_sim = ttk.Frame(notebook_graficas_sim)
    frame_random_sim = ttk.Frame(notebook_graficas_sim)

    notebook_graficas_sim.add(frame_regresion_sim, text="Regresión Lineal Múltiple")
    notebook_graficas_sim.add(frame_random_sim, text="Random Forest Regressor")

    """frame_sim_output = ttk.LabelFrame(frame_simulacion, text="Resultado de la Simulación")
    frame_sim_output.grid(column=1, row=1, padx=10, pady=10, sticky="nsew")"""
        
    # ---------------------- GRÁFICAS SIMULACIÓN REGRESIÓN LINEAL ----------------------
    figS1, axS1 = plt.subplots(figsize=(5, 4))  # Histograma de residuales
    figS2, axS2 = plt.subplots(figsize=(5, 4))  # Residuales por patrón
    figS3, axS3 = plt.subplots(figsize=(5, 4))  # Dispersión Yd vs Yr

    # Histograma de residuales
    axS1.set_title("Distribución de Residuales (Simulación)")
    axS1.set_xlabel("Error (Yd - Yr)")
    axS1.set_ylabel("Frecuencia")
    axS1.text(0.5, 0.5, "Ejecuta la simulación\npara ver la distribución de errores",
            ha='center', va='center', transform=axS1.transAxes, fontsize=11)
    figS1.tight_layout()

    # Residuales por patrón
    axS2.set_title("Residuales por Patrón (Simulación)")
    axS2.set_xlabel("Índice de patrón")
    axS2.set_ylabel("Error (Yd - Yr)")
    axS2.text(0.5, 0.5, "Ejecuta la simulación\npara ver los residuales",
            ha='center', va='center', transform=axS2.transAxes, fontsize=11)
    figS2.tight_layout()

    # Dispersión Yd vs Yr
    axS3.set_title("Dispersión Yd vs Yr (Simulación)")
    axS3.set_xlabel("Salida Deseada (Yd)")
    axS3.set_ylabel("Salida Obtenida (Yr)")
    axS3.text(0.5, 0.5, "Ejecuta la simulación\npara ver la dispersión",
            ha='center', va='center', transform=axS3.transAxes, fontsize=11)
    figS3.tight_layout()

    canvasS1 = FigureCanvasTkAgg(figS1, master=frame_regresion_sim)
    canvasS2 = FigureCanvasTkAgg(figS2, master=frame_regresion_sim)
    canvasS3 = FigureCanvasTkAgg(figS3, master=frame_regresion_sim)

    canvasS1.get_tk_widget().grid(row=0, column=0, padx=5, pady=5)
    canvasS2.get_tk_widget().grid(row=0, column=1, padx=5, pady=5)
    canvasS3.get_tk_widget().grid(row=1, column=0, padx=5, pady=5)

    textSimulacion = tk.Text(frame_regresion_sim, wrap="word", width=60)
    textSimulacion.grid(column=1, row=1, padx=5, pady=5)

    # ---------------------- GRÁFICAS SIMULACIÓN RANDOM FOREST REGRESSOR ----------------------
    figSRF1, axSRF1 = plt.subplots(figsize=(5, 4))  # Histograma de residuales
    figSRF2, axSRF2 = plt.subplots(figsize=(5, 4))  # Residuales por patrón
    figSRF3, axSRF3 = plt.subplots(figsize=(5, 4))  # Dispersión Yd vs Yr

    # Histograma de residuales
    axSRF1.set_title("Distribución de Residuales (Simulación)")
    axSRF1.set_xlabel("Error (Yd - Yr)")
    axSRF1.set_ylabel("Frecuencia")
    axSRF1.text(0.5, 0.5, "Ejecuta la simulación\npara ver la distribución de errores",
            ha='center', va='center', transform=axSRF1.transAxes, fontsize=11)
    figSRF1.tight_layout()

    # Residuales por patrón
    axSRF2.set_title("Residuales por Patrón (Simulación)")
    axSRF2.set_xlabel("Índice de patrón")
    axSRF2.set_ylabel("Error (Yd - Yr)")
    axSRF2.text(0.5, 0.5, "Ejecuta la simulación\npara ver los residuales",
            ha='center', va='center', transform=axSRF2.transAxes, fontsize=11)
    figSRF2.tight_layout()

    # Dispersión Yd vs Yr
    axSRF3.set_title("Dispersión Yd vs Yr (Simulación)")
    axSRF3.set_xlabel("Salida Deseada (Yd)")
    axSRF3.set_ylabel("Salida Obtenida (Yr)")
    axSRF3.text(0.5, 0.5, "Ejecuta la simulación\npara ver la dispersión",
            ha='center', va='center', transform=axSRF3.transAxes, fontsize=11)
    figSRF3.tight_layout()

    canvasSRF1 = FigureCanvasTkAgg(figSRF1, master=frame_random_sim)
    canvasSRF2 = FigureCanvasTkAgg(figSRF2, master=frame_random_sim)
    canvasSRF3 = FigureCanvasTkAgg(figSRF3, master=frame_random_sim)

    canvasSRF1.get_tk_widget().grid(row=0, column=0, padx=5, pady=5)
    canvasSRF2.get_tk_widget().grid(row=0, column=1, padx=5, pady=5)
    canvasSRF3.get_tk_widget().grid(row=1, column=0, padx=5, pady=5)

    textSimulacionRF = tk.Text(frame_random_sim, wrap="word", width=60)
    textSimulacionRF.grid(column=1, row=1, padx=5, pady=5)

    return root, text_area

# Inicializar la aplicación
root, text_area = crear_ventana()
root.mainloop()