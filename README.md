# spatial-geomarketing-vgi-  Pipeline
Replication code and spatial datasets for Spatial Geomarketing and VGI Integration study.
This repository contains the replication code and spatial datasets for the study:
**"Spatial Geomarketing and Volunteered Geographic Information (VGI) Integration for Commercial Success Modeling"**.
 
## Repository Structure


```text
.
├── data/processed/             # Spatial Datasets (Urban land use & Cafeterias with VGI)
├── scripts/                    # Python Scripts for Spatial Modeling and Evaluation
├── requirements.txt            # Python Dependencies
└── README.md                   # Replication Guide
```

## Requirements & Installation

The code requires **Python 3.10+**. Clone this repository and install the dependencies:

```bash
git clone [https://github.com/your-username/spatial-geomarketing-vgi.git](https://github.com/your-username/spatial-geomarketing-vgi.git)
cd spatial-geomarketing-vgi
pip install -r requirements.txt
```

## Main Python Packages:
* `geopandas` / `shapely` (Vector spatial operations & projected CRS processing)
* `pandas` / `numpy` (Vectorized matrix operations & data wrangling)
* `scipy` (Statistical hypothesis testing & Spearman Rank Correlation)
* `matplotlib` (Evaluation visualization)

---

## Execution Pipeline

To reproduce the complete experiment, execute the Python scripts sequentially from the repository root:

### **Stage 1: Traditional Huff Spatial Model**
```bash
python scripts/Huff_Clasico.py
```
*Computes the baseline spatial interaction model using traditional distance-decay parameters ($\alpha, \beta$).*

### **Stage 2: Proposed VGI-Augmented Spatial Model**
```bash
python scripts/Huff_BA.py
```
*Executes the proposed spatial allocation model incorporating the VGI composite digital reputation weight ($W_j$) and exponent ($\gamma$).*

### **Stage 3: Statistical Evaluation & Model Comparison**
```bash
python scripts/Comparacion_evaluacion.py
```
*Evaluates model performance against empirical review counts ($y_{\text{real}}$) using Spearman's rank correlation ($\rho$) and outputs comparative metrics.*

---

## Citation & License

If you use this dataset or code, please cite the Zenodo archive:

> Carrillo-García, N., Moreno-Ibarra, M., & Contreras-Juárez, Y. (2026). *Spatial Geomarketing and VGI Integration Pipeline Dataset and Codebase* (v1.0.0). Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX

Distributed under the **MIT License** for code and **CC-BY 4.0** for data assets.
