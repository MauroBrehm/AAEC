from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "modules" / "diabetes.csv"

df_diabetes = pd.read_csv(CSV_PATH)
print(df_diabetes.head())  # Muestra los datos en una tabla
print(df_diabetes.info())  # Muestra las instancias presentes de cada atributo
print(df_diabetes.describe())#Muestra un resumen de las estadísticas de los atributos numéricos


