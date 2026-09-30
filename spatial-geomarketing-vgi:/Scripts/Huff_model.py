#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 12:18:25 2025

@author: norbertocarrillogarcia
"""
####### ("codigo_act" ILIKE '%722515%') AND ("tipo_asent" ILIKE '%l%') AND ("nomb_asent" ILIKE '%VILLA COYOACAN%') AND ("municipio" ILIKE '%Coyoacán%')
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

# unidades_hab = gpd.read_file('U_Hab.shp') 
# unidades_hab = gpd.read_file('uso_de_suelo_coyoacan_p.shp') 
# cafeterias = gpd.read_file('denue_coyoacan.shp') 


# Cargar shapefiles para 
unidades_hab = gpd.read_file('capas/uso_de_suelo/VC/uso_suelo_vc.shp') 
cafeterias = gpd.read_file('capas/cafeterias/VC/cafeterias_2016_vc.shp') 

# seleccionar los atributos de 
cafeterias = cafeterias[["id", "nom_estab", "per_ocu", "latitud","longitud", "geometry"]]

# area = unidades_hab.area
# area_total = sum(area)

area = unidades_hab["superfc"]
area_total = sum(area)

def calcular_distancias(unidades_hab, cafeterias):
    return cafeterias.geometry.apply(lambda x: unidades_hab.distance(x, align=True))

def mapear_intervalo(intervalo):
    palabras = intervalo.replace(' personas', '').split(' a ')
    if len(palabras) == 2:
        return ((int(palabras[0]) + int(palabras[1])) / 2)
    else:
        raise ValueError(f"Formato de intervalo no reconocido: {intervalo}")
        
unidades_hab = unidades_hab.to_crs(epsg=3857)
cafeterias = cafeterias.to_crs(epsg=3857)

distancias = calcular_distancias(unidades_hab, cafeterias)
cafeterias["atraccion"] = cafeterias['per_ocu'].apply(mapear_intervalo)

utilidades_totales = pd.DataFrame()

for x in range(distancias.shape[1]):
    utilidades_totales[x] = cafeterias["atraccion"]/(distancias[x])**2
    
probabilidades_totales = pd.DataFrame()


# print(utilidades_totales[0])
for j in range(utilidades_totales.shape[1]):
    suma =  sum(utilidades_totales[j])
    probabilidades_totales[j] = utilidades_totales[j]/suma
    

# print(utilidades_totales[0])
for j in range(utilidades_totales.shape[1]):
    suma =  sum(utilidades_totales[j])
    probabilidades_totales[j] = (utilidades_totales[j]/suma) * 0.3
    
    
prob_tot = []
# for z in range(probabilidades_totales.shape[0]):
#     prob_tot.append(sum(probabilidades_totales.iloc[z]))   
    
for z in range(probabilidades_totales.shape[0]):
    pivot = sum((probabilidades_totales.iloc[z] * area) / area_total) 
    prob_tot.append(pivot)
# hasta aquí, todo bien solo que se necesita la poblacion de cada uno de los vecindarios 
prob = pd.DataFrame(prob_tot)
porciento = round(prob,4)* 100

cafeterias["prob_pond"] = prob
cafeterias["porciento"] = porciento

# print(round(prob,4)* 100)
cafeterias.to_file('cafeterias_prob_pond_vc.shp')
print(cafeterias.crs)

cafeterias_2 = cafeterias

cafeterias_2["prob_pond"] = cafeterias_2["prob_pond"] * random.uniform(0, 0.5)

cafeterias.to_file('outputs/cafeterias_prob_pond_vc_2.shp')
print(cafeterias.crs)