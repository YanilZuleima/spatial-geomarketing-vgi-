#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 14:57:03 2026

@author: norbertocarrillogarcia
"""

import geopandas as gpd
import pandas as pd
import numpy as np

# ==============================================================================
# CONFIGURACIÓN DE HIPERPARÁMETROS DOCTORALES
# ==============================================================================
alfa = 0.5  # Sensibilidad al atractivo físico (S_j^alfa)
beta = 0.5  # Fricción de la distancia geográfica (T_ij^beta)
gama = 2.0  # Control del peso digital VGI (W_j^gama)
            # 0 = Huff Tradicional puro
            # 1 = Modelo compuesto base
            # >1 = Hipersensibilidad a la reputación online

# 1. CARGA Y FILTRADO DE DATOS
unidades_hab = gpd.read_file('capas/uso_de_suelo/VC/uso_suelo_vc.shp') 
cafeterias = gpd.read_file('capas/cafeterias/VC/cafeterias_cuauhtemoc_con_reseñas.shp') 
#cafeterias["w_vgi"] = np.random.uniform(1.0, 5.0, len(cafeterias))
# Seleccionar atributos base + tu columna de ponderación preprocesada de reseñas
cafeterias = cafeterias[["id", "nom_estab", "per_ocu", "w_vgi", "latitud", "longitud", "geometry"]]

# Forzar proyección métrica (EPSG:3857) para cálculo preciso de distancias
unidades_hab = unidades_hab.to_crs(epsg=3857)
cafeterias = cafeterias.to_crs(epsg=3857)

area = unidades_hab["superfc"].values
area_total = area.sum()

def calcular_distancias(unidades_hab, cafeterias):
    return cafeterias.geometry.apply(lambda x: unidades_hab.distance(x, align=True))

# 2. PROCESAMIENTO DEL ATRACTIVO FÍSICO BASE (S_j)
def mapear_intervalo(intervalo):
    palabras = str(intervalo).replace(' personas', '').split(' a ')
    if len(palabras) == 2:
        return ((int(palabras[0]) + int(palabras[1])) / 2)
    return 5.0

cafeterias["S_j"] = cafeterias['per_ocu'].apply(mapear_intervalo)

# 3. MATRIZ DE DISTANCIAS BASE
distancias_matriz = calcular_distancias(unidades_hab, cafeterias).values
distancias_matriz = np.where(distancias_matriz == 0, 1.0, distancias_matriz)

# ==============================================================================
# CÁLCULO MATRICIAL DEL MODELO PROPUESTO (CON ALFA, BETA Y GAMA)
# ==============================================================================

# Paso A: Componente Espacial de Huff -> (S_j^alfa) / (T_ij^beta)
atraccion_con_alfa = cafeterias["S_j"].values[:, np.newaxis] ** alfa
distancia_con_beta = distancias_matriz ** beta
utilidades_espaciales = atraccion_con_alfa / distancia_con_beta

# Convertir a Probabilidad Espacial (Normalizado por columna)
prob_espacial = utilidades_espaciales / utilidades_espaciales.sum(axis=0)

# Paso B: Componente Digital Afectado por GAMA -> W_j^gama
# Aplicamos el exponente gama al vector de pesos VGI
pesos_con_gama = cafeterias["w_vgi"].values ** gama
suma_pesos_gama = pesos_con_gama.sum()

# Calcular P_wij afectado por gama (vector columna para broadcasting)
prob_vgi_vector = (pesos_con_gama / suma_pesos_gama)[:, np.newaxis]

# Paso C: Probabilidad Compuesta Cruzada -> P_ij = Prob_Espacial * Prob_VGI
P_ij_compuesta = prob_espacial * prob_vgi_vector

# RE-NORMALIZACIÓN CRÍTICA: Garantizar que cada columna (polígono i) sume exactamente 1
P_ij_final = P_ij_compuesta / P_ij_compuesta.sum(axis=0)

# ==============================================================================
# AGREGACIÓN TERRITORIAL (Asignación de Demanda sobre el Terreno)
# ==============================================================================
# Multiplicamos la matriz de probabilidades por el vector de áreas de los vecindarios
demanda_por_cafeteria = (P_ij_final * area).sum(axis=1)

# Guardar métricas en el DataFrame de salida
cafeterias["demanda_est"] = demanda_por_cafeteria
cafeterias["porciento_mkt"] = (demanda_por_cafeteria / area_total) * 100

# 4. EXPORTACIÓN DE RESULTADOS
cafeterias.to_file('outputs/cafeterias_modelo_zule_completo.shp')
print(f"¡Modelo ejecutado con éxito!")
print(f"Parámetros utilizados -> alfa: {alfa}, beta: {beta}, gama: {gama}")
print(cafeterias[["nom_estab", "w_vgi", "demanda_est", "porciento_mkt"]].head())