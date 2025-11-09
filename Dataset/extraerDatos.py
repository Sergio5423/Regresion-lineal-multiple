import pandas as pd
import numpy as np
import random

# Semilla para reproducibilidad
np.random.seed(42)

# Barrios simulados con estratos asociados
barrios_estratos = {
    "La Nevada": 1,
    "Los Fundadores": 2,
    "Villa Miriam": 3,
    "Novalito": 4,
    "Los Cortijos": 5,
    "Rosario": 6
}

# Generador de datos
def generar_datos(n=1000):
    barrios = list(barrios_estratos.keys())
    datos = []

    for _ in range(n):
        barrio = random.choice(barrios)
        estrato = barrios_estratos[barrio]
        area = np.random.normal(loc=80 + estrato * 10, scale=15)  # más estrato → más área
        habitaciones = np.random.poisson(lam=2 + estrato * 0.3)
        baños = max(1, int(habitaciones / 2))
        antiguedad = np.random.randint(0, 40)
        parqueadero = int(np.random.rand() < 0.6 if estrato >= 4 else np.random.rand() < 0.3)
        ascensor = int(np.random.rand() < 0.7 if estrato >= 5 else np.random.rand() < 0.2)
        cercania = np.random.normal(loc=600 - estrato * 50, scale=100)
        desempleo = np.random.normal(loc=0.15 - estrato * 0.01, scale=0.02)

        # Modelo de precio (COP)
        precio = (
            1_000_000 * area +
            15_000_000 * habitaciones +
            10_000_000 * baños +
            20_000_000 * estrato +
            5_000_000 * parqueadero +
            8_000_000 * ascensor -
            500_000 * antiguedad -
            10_000 * cercania +
            np.random.normal(0, 20_000_000)  # ruido
        )

        datos.append({
            "precio": int(precio),
            "area": round(area, 1),
            "habitaciones": habitaciones,
            "baños": baños,
            "antiguedad": antiguedad,
            "estrato": estrato,
            "parqueadero": parqueadero,
            "ascensor": ascensor,
            "barrio": barrio,
            "cercania_equipamientos": round(cercania, 1),
            "tasa_desempleo_sector": round(desempleo, 3)
        })

    return pd.DataFrame(datos)

# Generar y guardar
df = generar_datos(1000)
df.to_csv("datos_sinteticos_correlacionados.csv", index=False)
print("Datos sintéticos generados con correlaciones realistas.")
