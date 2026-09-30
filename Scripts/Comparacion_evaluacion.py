#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 15:17:56 2026

@author: norbertocarrillogarcia
"""
import geopandas as gpd
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# 1. CARGAR LOS TRES ESCENARIOS GENERADOS
huff_clasico = gpd.read_file('outputs/cafeterias_prob_pond_vc_2.shp')
modelo_propuesto = gpd.read_file('outputs/cafeterias_modelo_zule_completo.shp')
machine_learning = gpd.read_file('outputs/cafeterias_modelo_machine_learning.shp')
#machine_learning["num_reseñas_real"] = np.random.randint(10, 1000, len(machine_learning))
voronoi = gpd.read_file('outputs/cafeterias_modelo_voronoi.shp')

col_voronoi = [c for c in voronoi.columns if 'porciento' in c or 'mkt' in c][0]
score_voronoi = voronoi[col_voronoi].values

# 2. IDENTIFICAR LAS COLUMNAS CORRECTAS
col_huff = [c for c in huff_clasico.columns if 'porciento' in c or 'prob' in c][0]
col_propuesto = [c for c in modelo_propuesto.columns if 'porciento' in c or 'mkt' in c][0]
col_ml = [c for c in machine_learning.columns if 'porciento' in c or 'mkt' in c][0]

# Extraer los vectores de predicción (Scores)
score_huff = huff_clasico[col_huff].values
score_propuesto = modelo_propuesto[col_propuesto].values
score_ml = machine_learning[col_ml].values

# 3. EXTRAER LA REALIDAD (El volumen total de reseñas reales de tu capa)
# Usamos la columna de la capa de Machine Learning o de donde tengas guardado el dato real
y_real_continuo = machine_learning['num_res_re'].values

# ==============================================================================
# CÁLCULO RIGOROSO DE CORRELACIÓN DE SPEARMAN
# ==============================================================================
rho_huff, p_huff = stats.spearmanr(score_huff, y_real_continuo)
rho_propuesto, p_propuesto = stats.spearmanr(score_propuesto, y_real_continuo)
rho_ml, p_ml = stats.spearmanr(score_ml, y_real_continuo)
rho_voronoi, p_voronoi = stats.spearmanr(score_voronoi, y_real_continuo)

# Reemplazar NaNs si la muestra es tan pequeña que la varianza es cero
rho_huff = 0.0 if np.isnan(rho_huff) else rho_huff
rho_propuesto = 0.0 if np.isnan(rho_propuesto) else rho_propuesto
rho_ml = 0.0 if np.isnan(rho_ml) else rho_ml

# ==============================================================================
# REPORTE DE RESULTADOS PARA LA TESIS
# ==============================================================================
print("=== REPORTE DE EVALUACIÓN GEOMARKETING (RANGOS DE SPEARMAN) ===")
print(f"Modelos evaluados sobre una muestra de {len(y_real_continuo)} unidades activas.\n")
print(f"1. Huff Tradicional Puro:         Rho = {rho_huff:.4f} (p-value: {p_huff})")
print(f"2. Tu Modelo Propuesto (VGI):     Rho = {rho_propuesto:.4f} (p-value: {p_propuesto})")
print(f"3. Machine Learning (XGBoost):    Rho = {rho_ml:.4f} (p-value: {p_ml})")
print(f"4. Nearest Facility (Voronoi):   Rho = {rho_voronoi:.4f} (p-value: {p_voronoi})")

# ==============================================================================
# GRÁFICA COMPARATIVA DE BARRAS
# ==============================================================================
plt.figure(figsize=(8, 5))
modelos = ['Huff Tradicional', 'Tu Modelo (Huff+VGI)', 'Machine Learning (XGBoost)']
valores_rho = [rho_huff, rho_propuesto, rho_ml]

barras = plt.bar(modelos, valores_rho, color=['orange', 'blue', 'green'], alpha=0.8, width=0.5)

# Añadir los valores numéricos arriba de cada barra
for barra in barras:
    yval = barra.get_height()
    plt.text(barra.get_x() + barra.get_width()/2, yval + 0.02, f'{yval:.4f}', ha='center', va='bottom', fontweight='bold')

plt.ylim(-1.1, 1.1)
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.ylabel('Coeficiente de Spearman (Rho)', fontsize=12)
plt.title('Comparativa de Modelos: Capacidad de Ordenamiento del Éxito Comercial', fontsize=13, pad=15)
plt.grid(axis='y', linestyle=':', alpha=0.6)

plt.savefig('outputs/comparativa_spearman_modelos.png', dpi=300, bbox_inches='tight')
plt.show()