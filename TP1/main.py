from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

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
#print(df_diabetes.head())  # Muestra los datos en una tabla
# print(df_diabetes.info())  # Muestra las instancias presentes de cada atributo
# print(df_diabetes.describe())#Muestra un resumen de las estadísticas de los atributos numéricos


columnas=[ "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI",
           "DiabetesPedigreeFunction", "Age", "Outcome" ]
# for i in columnas:
#     df_diabetes[i].hist(bins=50, edgecolor='k', figsize=(20,15))
#     plt.title(i)
#     plt.show()

#Actividad 3 Division del dataser en conjunto de entrenamiento y de prueba
#Eliminamos la primera columna 
if 'Unnamed: 0' in df_diabetes.columns:
    df_diabetes = df_diabetes.drop(columns=['Unnamed: 0'])
#Remplazamos aquellos valores que No pueden ser 0
cols_con_ceros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

df_diabetes_nan = df_diabetes.copy()
df_diabetes_nan[cols_con_ceros] = df_diabetes_nan[cols_con_ceros].replace(0, np.nan)
#Ahora dividimos el dataset en conjunto de entrenamiento y de validación
X = df_diabetes_nan.drop(columns=['Outcome'])
y = df_diabetes_nan['Outcome']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Imputación mediante KNNImputer (Ajustado SOLO con X_train)
imputer = KNNImputer(n_neighbors=5)

# Entrenar el imputador en Train y transformar
X_train_imputed = pd.DataFrame(
    imputer.fit_transform(X_train), 
    columns=X_train.columns, 
    index=X_train.index
)

#Reconstruimos el dataset de entrenamiento imputado con la variable Outcome
train_imputed = X_train_imputed.copy()
train_imputed['Outcome'] = y_train

#Vamos a realizar la matriz de correlacion para ver la relacion entre las variables
plt.figure(figsize=(10, 8))
sns.heatmap(train_imputed.corr(), annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title("Matriz de Correlación")
plt.show()

#Actividad 5(mostrar la distribucion de la variable outcome)
# print(df_diabetes["Outcome"].value_counts())
# print(df_diabetes["Outcome"].describe())
