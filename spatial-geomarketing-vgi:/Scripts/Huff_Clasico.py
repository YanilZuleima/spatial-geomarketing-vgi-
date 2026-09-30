#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 12:27:44 2026

@author: norbertocarrillogarcia
"""

import geopandas as gpd
import pandas as pd
import numpy as np

# ==========================================
# CONFIGURACIÓN DE HIPERPARÁMETROS (HUFF CLÁSICO)
# ==========================================
alfa = 0.5  # Parámetro para el atractivo físico (atraccion^alfa)
beta = 0.5  # Parámetro para la fricción de la distancia (distancia^beta)

# Cargar shapefiles 
unidades_hab = gpd.read_file('capas/uso_de_suelo/VC/uso_suelo_vc.shp') 
cafeterias = gpd.read_file('capas/cafeterias/VC/cafeterias_cuauhtemoc_con_reseñas.shp') 

# Seleccionar los atributos
cafeterias = cafeterias[["id", "nom_estab", "per_ocu", "latitud","longitud", "geometry"]]

area = unidades_hab["superfc"].values
area_total = area.sum()

def calcular_distancias(unidades_hab, cafeterias):
    return cafeterias.geometry.apply(lambda x: unidades_hab.distance(x, align=True))
        
unidades_hab = unidades_hab.to_crs(epsg=3857)
cafeterias = cafeterias.to_crs(epsg=3857)

# 1. Matriz de distancias base (Num_Cafeterias, Num_Poligonos)
distancias_matriz = calcular_distancias(unidades_hab, cafeterias).values
distancias_matriz = np.where(distancias_matriz == 0, 1.0, distancias_matriz)

def mapear_intervalo(intervalo):
    palabras = str(intervalo).replace(' personas', '').split(' a ')
    if len(palabras) == 2:
        return ((int(palabras[0]) + int(palabras[1])) / 2)
    else:
        return 5.0

cafeterias["atraccion"] = cafeterias['per_ocu'].apply(mapear_intervalo)

# Vector de atracción base en formato columna (N, 1)
atraccion_base = cafeterias["atraccion"].values[:, np.newaxis]

# =========================================================
# APLICACIÓN DE ALFA Y BETA (Estructura de tu script)
# =========================================================

# Utilidad = (Atraccion^alfa) / (Distancia^beta)
utilidades_matriz = (atraccion_base ** alfa) / (distancias_matriz ** beta)

# Convertir a probabilidades (Normalizado por columna / polígono)
suma_utilidades = utilidades_matriz.sum(axis=0)
probabilidades_matriz = utilidades_matriz / suma_utilidades

# Tu factor de ajuste original del script (* 0.3)
probabilidades_matriz_ajustada = probabilidades_matriz * 0.3

# Ponderación por área (Agregación territorial de tu script)
prob_ponderada_por_area = probabilidades_matriz_ajustada * area
prob_tot = prob_ponderada_por_area.sum(axis=1) / area_total

# Guardar resultados en el DataFrame
cafeterias["prob_pond"] = prob_tot
cafeterias["porciento"] = np.round(prob_tot, 4) * 100

# Exportar primer escenario
cafeterias.to_file('cafeterias_prob_pond_vc.shp')
print("Primer archivo guardado. CRS:", cafeterias.crs)

# Generación del escenario aleatorio secundario de tu script
factor_aleatorio = np.random.uniform(0, 0.5)
cafeterias_2 = cafeterias.copy()
cafeterias_2["prob_pond"] = cafeterias_2["prob_pond"] * factor_aleatorio
cafeterias_2["porciento"] = np.round(cafeterias_2["prob_pond"], 4) * 100

# Exportar segundo escenario
cafeterias_2.to_file('outputs/cafeterias_prob_pond_vc_2.shp')
print("Segundo archivo guardado. CRS:", cafeterias_2.crs)
print(f"Huff Tradicional ejecutado con éxito (alfa={alfa}, beta={beta})")