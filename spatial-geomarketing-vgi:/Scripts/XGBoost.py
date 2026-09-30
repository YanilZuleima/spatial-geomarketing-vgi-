#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 15:02:32 2026

@author: norbertocarrillogarcia
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score

# 1. CARGA DE DATOS Y PROYECCIÓN
unidades_hab = gpd.read_file('capas/uso_de_suelo/VC/uso_suelo_vc.shp') 
cafeterias = gpd.read_file('capas/cafeterias/VC/cafeterias_cuauhtemoc_con_reseñas.shp') 
#cafeterias["w_vgi"] = np.random.uniform(1.0, 5.0, len(cafeterias))
#cafeterias["num_reseñas_real"] = np.random.randint(10, 1000, len(cafeterias))
# Asegúrate de tener: 'w_vgi' (peso) y 'num_reseñas_real' (el total de comentarios/proxy real)
cafeterias = cafeterias[["id", "nom_estab", "per_ocu", "w_vgi", "num_res_re", "geometry"]]

unidades_hab = unidades_hab.to_crs(epsg=3857)
cafeterias = cafeterias.to_crs(epsg=3857)

area = unidades_hab["superfc"].values

# 2. EXTRACCIÓN DE ATRACTIVO FÍSICO BASE (S_j)
def mapear_intervalo(intervalo):
    palabras = str(intervalo).replace(' personas', '').split(' a ')
    if len(palabras) == 2:
        return ((int(palabras[0]) + int(palabras[1])) / 2)
    return 5.0

cafeterias["S_j"] = cafeterias['per_ocu'].apply(mapear_intervalo)

# 3. CÁLCULO DE LA MATRIZ DE DISTANCIAS
distancias_matriz = cafeterias.geometry.apply(lambda x: unidades_hab.distance(x, align=True)).values
distancias_matriz = np.where(distancias_matriz == 0, 1.0, distancias_matriz)

# ==============================================================================
# INGENIERÍA DE CARACTERÍSTICAS GEOESPACIALES (Para la IA)
# ==============================================================================
# Vamos a crear métricas del entorno para cada cafetería (filas de la matriz)

# Feature 1: Distancia promedio a todas las unidades habitacionales
dist_promedio = distancias_matriz.mean(axis=1)

# Feature 2: Distancia al vecindario más cercano (accesibilidad inmediata)
dist_minima = distancias_matriz.min(axis=1)

# Feature 3: Masa residencial circundante ponderada por el inverso de la distancia
# (Un indicador de cuánta población potencial tiene cerca físicamente)
masa_cercana = (area / distancias_matriz).sum(axis=1)

# Construir el DataFrame de características (X) y el objetivo (y)
X = pd.DataFrame({
    'S_j': cafeterias['S_j'].values,          # Atractivo físico (DENUE)
    'w_vgi': cafeterias['w_vgi'].values,      # Atractivo digital (Tu procesamiento RoBERTa)
    'dist_promedio': dist_promedio,
    'dist_minima': dist_minima,
    'masa_cercana': masa_cercana
})

# Nuestro Target 'y' es el volumen total de reseñas reales de la cafetería
y = cafeterias['num_res_re'].values

# ==============================================================================
# ENTRENAMIENTO DEL MODELO DE MACHINE LEARNING
# ==============================================================================
# Dividimos en conjunto de entrenamiento y prueba (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializar y entrenar el regresor XGBoost
# Usamos hiperparámetros estándar bien balanceados para evitar sobreajuste
ml_model = XGBRegressor(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    random_state=42
)

ml_model.fit(X_train, y_train)

# EVALUACIÓN LOCAL DEL MODELO DE IA
y_pred_test = ml_model.predict(X_test)
rmse_test = root_mean_squared_error(y_test, y_pred_test)
r2_test = r2_score(y_test, y_pred_test)

print("--- RENDIMIENTO DEL MODELO XGBOOST (CONJUNTO DE PRUEBA) ---")
print(f"RMSE: {rmse_test:.4f}")
print(f"R² Score: {r2_test:.4f}\n")

# ==============================================================================
# PREDICCIÓN GLOBAL Y EXPORTACIÓN ASIGNADA
# ==============================================================================
# Predecir para TODAS las cafeterías del mapa
predicciones_globales = ml_model.predict(X)

# Forzar a que no existan predicciones negativas (físicamente imposible)
predicciones_globales = np.clip(predicciones_globales, 0, None)

# Guardar los resultados en el GeoDataFrame
cafeterias["demanda_est"] = predicciones_globales
# Convertir a participación porcentual para que empate con las métricas de tus otros scripts
cafeterias["porciento_mkt"] = (predicciones_globales / predicciones_globales.sum()) * 100

# Exportar la capa final de Machine Learning
cafeterias.to_file('outputs/cafeterias_modelo_machine_learning.shp')
print("¡Modelo de Machine Learning ejecutado y guardado en 'outputs/cafeterias_modelo_machine_learning.shp'!")

# Mostrar la importancia de las variables (Feature Importance) para tu texto de tesis
importancias = pd.DataFrame({
    'Variable': X.columns,
    'Importancia': ml_model.feature_importances_
}).sort_values(by='Importancia', ascending=False)

print("\n--- IMPORTANCIA DE LAS VARIABLES SEGÚN XGBOOST ---")
print(importancias.to_string(index=False))