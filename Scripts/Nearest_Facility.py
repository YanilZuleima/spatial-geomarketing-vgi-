#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 17:01:44 2026

@author: norbertocarrillogarcia
"""

import geopandas as gpd
import pandas as pd
import numpy as np

# 1. CARGA Y FILTRADO DE DATOS
unidades_hab = gpd.read_file('capas/uso_de_suelo/VC/uso_suelo_vc.shp') 
cafeterias = gpd.read_file('capas/cafeterias/VC/cafeterias_cuauhtemoc_con_reseñas.shp') 

# Seleccionar los atributos esenciales
cafeterias = cafeterias[["id", "nom_estab", "per_ocu", "latitud", "longitud", "geometry"]]

# Transformación de coordenadas a sistema métrico (EPSG:3857)
unidades_hab = unidades_hab.to_crs(epsg=3857)
cafeterias = cafeterias.to_crs(epsg=3857)

# Extraer el vector de áreas residenciales
area = unidades_hab["superfc"].values
area_total = area.sum()

# 2. CÁLCULO DE LA MATRIZ DE DISTANCIAS (Num_Cafeterias, Num_Poligonos)
distancias_matriz = cafeterias.geometry.apply(lambda x: unidades_hab.distance(x, align=True)).values

# ==============================================================================
# ALGORITMO NEAREST FACILITY (ASIGNACIÓN DETERMINISTA DE TODO O NADA)
# ==============================================================================

# Paso A: Encontrar el índice de la cafetería más cercana para cada polígono i
# argmin busca a lo largo del eje 0 (columnas), dándonos el renglón (cafetería) ganador
indice_cafeteria_mas_cercana = np.argmin(distancias_matriz, axis=0)

# Paso B: Crear una matriz de asignación binaria (ceros y unos)
# Tendrá el mismo tamaño que la matriz de distancias
matriz_asignacion = np.zeros_like(distancias_matriz)

# Marcamos con un 1 a la cafetería ganadora para cada unidad habitacional
matriz_asignacion[indice_cafeteria_mas_cercana, np.arange(len(indice_cafeteria_mas_cercana))] = 1.0

# Paso C: Captura del mercado (Multiplicar la asignación rígida por el área)
area_capturada_por_cafeteria = (matriz_asignacion * area).sum(axis=1)

# ==============================================================================
# ASIGNACIÓN DE RESULTADOS Y EXPORTACIÓN
# ==============================================================================
cafeterias["demanda_est"] = area_capturada_por_cafeteria
cafeterias["porciento_mkt"] = (area_capturada_por_cafeteria / area_total) * 100

# Exportar la capa resultante para tu validador
cafeterias.to_file('outputs/cafeterias_modelo_voronoi.shp')
print("¡Modelo de Nearest Facility (Voronoi) ejecutado y guardado con éxito!")
print(cafeterias[["nom_estab", "demanda_est", "porciento_mkt"]].head())