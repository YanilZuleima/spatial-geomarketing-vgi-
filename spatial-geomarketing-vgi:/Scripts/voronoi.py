#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 12:51:01 2026

@author: norbertocarrillogarcia
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from shapely.geometry import LineString
from geovoronoi import voronoi_regions_from_coords
from geovoronoi import points_to_coords
from shapely.geometry import Polygon

# =====================================
# CARGAR DATOS
# =====================================

cafeterias = gpd.read_file('capas/cafeterias/VC/cafeterias_2016_vc.shp') 
unidades_hab = gpd.read_file('capas/uso_de_suelo/VC/uso_suelo_vc.shp')
cafeterias = cafeterias[["id", "nom_estab", "per_ocu", "latitud","longitud", "geometry"]]




# =====================================
# CRS
# =====================================

unidades_hab = unidades_hab.to_crs(
    epsg=3857
)

cafeterias = cafeterias.to_crs(
    epsg=3857
)

# centroides

centroides = unidades_hab.copy()

centroides["geometry"] = centroides.centroid

asignacion = []

for idx, punto in centroides.iterrows():

    distancias = cafeterias.distance(

        punto.geometry

    )

    idx_min = distancias.idxmin()

    asignacion.append(idx_min)

centroides["cafeteria"] = asignacion
resultado = (

    centroides

    .groupby("cafeteria")["superfc"]

    .sum()

)
resultado = resultado.reset_index()

resultado["prob_pond"] = (

    resultado["superfc"]

    /

    resultado["superfc"].sum()

)

resultado["porcentaje"] = (

    resultado["prob_pond"] * 100

)
cafeterias["cafeteria"] = cafeterias.index

cafeterias = cafeterias.merge(

    resultado,

    on="cafeteria",

    how="left"

)

fig, ax = plt.subplots(figsize=(12,12))

centroides.plot(

    column="cafeteria",

    categorical=True,

    legend=True,

    ax=ax

)

cafeterias.plot(

    ax=ax,

    color="black",

    markersize=50

)

plt.show()

lineas = []

for idx, row in centroides.iterrows():

    cafe = cafeterias.loc[row["cafeteria"]]

    linea = LineString([

        row.geometry,

        cafe.geometry

    ])

    lineas.append(linea)

lineas_gdf = gpd.GeoDataFrame(

    geometry=lineas,

    crs=centroides.crs

)

lineas_gdf.to_file(

    "lineas_nearest_facility.shp"

)