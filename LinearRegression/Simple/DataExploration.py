import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np

#Leer dataset
data = pd.read_csv("Datasets/FuelConsumptionCo2.csv")

#Ver algo del contenido
#print(data.head()) #Por defecto head() retorna los 5 primeros registros

#EXPLORACIÓN DE LOS DATOS
#Resumir los datos
#print(data.describe()) #con describe() se obtienen los percentiles 25, 50 y 75 por defecto, usando las columnas numéricas.
#genera estadísticas como el número de datos, promedio o media, desviación estandar, valor mínimo y máximo y el percentil o percentiles por defecto.

#Seleccionar algunas características para explorar más
#Elegir las columnas del dataset de interés
selData = data[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
#Mostrar los primeros 9 registros
print(selData.head(9))
#graficar los registros en un histograma
#Se puede usar selData para graficar pero si se desea otro orden a mostrar se hace lo siguiente
plotsData = selData[['CYLINDERS','ENGINESIZE','CO2EMISSIONS','FUELCONSUMPTION_COMB']]
#crear el histograma para cada columna del dataset
plotsData.hist()
plt.show()

#graficar cada una de las características vs la emisiónCo2 para ver que tan lineal es su relacón.
plt.scatter(selData.FUELCONSUMPTION_COMB, selData.CO2EMISSIONS, color='blue') 
#scatter permite crear graficos propios
plt.xlabel("Consumo de combustible comb")
plt.ylabel("Emisión de CO2")
plt.show()

plt.scatter(selData.ENGINESIZE, selData.CO2EMISSIONS, color='blue') 
plt.xlabel("Tamaño del motor")
plt.ylabel("Emisión de CO2")
plt.show()

plt.scatter(selData.CYLINDERS, selData.CO2EMISSIONS, color='blue') 
plt.xlabel("Cilindros")
plt.ylabel("Emisión de CO2")
plt.show()