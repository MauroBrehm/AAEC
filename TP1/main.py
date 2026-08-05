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

df_diabetes.hist(bins=50, edgecolor='k', figsize=(20,15))
plt.show()
columnas=[ "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
           "DiabetesPedigreeFunction", "Age", "Outcome" ]
for i in columnas:
    df_diabetes[i].hist(bins=50, edgecolor='k', figsize=(20,15))
    plt.title(i)
    plt.show()

#Actividad 3 Division del dataser en conjunto de entrenamiento y de prueba
#Vamos a realizar la matriz de correlacion para ver la relacion entre las variables
print(df_diabetes.corr())


#Actividad 5(mostrar la distribucion de la variable outcome)
print(df_diabetes["Outcome"].value_counts())

