"""
Para este caso se usará la división del dataset en Train/Test, esto se hará de tal forma que el 80%
del dataset quede para entrenar el modelo y el 20% restante para hacer las pruebas. 
Esta forma de entrenamiento permite que el modelo quede mucho más preciso, ya que no se usarán los 
datos que se usaron para su entrenamiento, lo que permite que sea más fiable y realista para el
problema que se soluciona en el mundo real.

La regresión lineal se ajusta a un modelo lineal con coeficientes B = (B1, ..., Bn) para minimizar 
la 'suma residual de cuadrados' entre la X independiente en el conjunto de datos y la Y dependiente 
por la aproximación lineal.
"""

import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score

#Leer dataset
data = pd.read_csv("Datasets/FuelConsumptionCo2.csv")
#Elegir las columnas del dataset de interés
selData = data[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
#Dividir el dataset para Train/Test y filtrar por las columnas de interés
mask = np.random.rand(len(data)) < 0.8 #Seleccionar las filas aleatoriamente para entrenar el modelo
dataTrain = selData[mask]
dataTest = selData[~mask]

#Distribución de los datos de entrenamiento
plt.scatter(dataTrain.ENGINESIZE, dataTrain.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
#plt.show()

#Modelado
#Definir el tipo de regresión
regression = linear_model.LinearRegression()
#Definir la variable dependiente Y y la independiente X
dataTrainX = np.asanyarray(dataTrain[['ENGINESIZE']])
dataTrainY = np.asanyarray(dataTrain[['CO2EMISSIONS']])
#Ajustar y entrenar el modelo lineal con los datos
regression.fit(dataTrainX, dataTrainY)
#Obtener el valor de los coeficientes obtenidos 
print("Pendiente: ", regression.coef_)
print("Punto de corte: ", regression.intercept_)

#Graficar la recta ajustada obtenida
plt.scatter(dataTrain.ENGINESIZE, dataTrain.CO2EMISSIONS,  color='blue')
plt.plot(dataTrainX, regression.coef_[0][0]*dataTrainX + regression.intercept_[0], '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
#plt.show()

#Evaluación del modelo
dataTestX = np.asanyarray(dataTest[['ENGINESIZE']])
dataTestY = np.asanyarray(dataTest[['CO2EMISSIONS']])
dataMTestY = regression.predict(dataTestX)

print("Mean absolute error: %.2f" % np.mean(np.absolute(dataMTestY - dataTestY)))
print("Mean Squared Error (MSE): %.2f" % np.mean((dataMTestY - dataTestY) ** 2))
print("R2-score: %.2f" % r2_score(dataMTestY , dataTestY))