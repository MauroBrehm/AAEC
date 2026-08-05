from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Desactivar el límite de columnas al imprimir en la terminal
pd.set_option('display.max_columns', None)

# Evitar que Pandas acorte el ancho total de la consola
pd.set_option('display.width', 1000)


BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "modules" / "diabetes.csv"
#Les conviene cometar aquellas lineas que no quieran ver lo que ejecuten, por las
#dudas para comentar varias lineas a la ves las seleccionan y ponen ctrl+k+c y para
#descomentar ctrl+k+u
df_diabetes = pd.read_csv(CSV_PATH)
print(df_diabetes.head())  # Muestra los datos en una tabla
print(df_diabetes.info())  # Muestra las instancias presentes de cada atributo
print(df_diabetes.describe())#Muestra un resumen de las estadísticas de los atributos numéricos
print(df_diabetes["Glucose"].value_counts())
print(df_diabetes["BloodPressure"].value_counts())
print(df_diabetes["SkinThickness"].value_counts())
print(df_diabetes["Insulin"].value_counts())
print(df_diabetes["BMI"].value_counts())
print(df_diabetes["DiabetesPedigreeFunction"].value_counts())
df_diabetes.hist(bins=50, edgecolor='k', figsize=(20,15))
plt.show()



