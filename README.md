
# 5G_Orch — Resource Management & Orchestration in 5G (UMRO-5G) + Simulation Suite

Este directorio contiene el material base del artículo (formato tipo revista) y un conjunto de simulaciones en Python que reproducen y apoyan resultados numéricos relacionados con **gestión de recursos y orquestación en redes 5G**, incluyendo slicing, planificación (scheduling), latencia en SFC, convergencia DRL y comparación de complejidad computacional.

---

## 1) Contenido del directorio

### Artículo / secciones (Markdown)
- **`Management_Orchestration_5G_Resources_MDPI_Computation.md`**  
  Manuscrito principal en formato *MDPI Computation* (Review/Survey). Incluye el marco unificado y la propuesta **UMRO-5G**, además de la sección de validación numérica (simulaciones).
- **`Management_Orchestration_5G_AI_Native_6G_IEEE.md`**  
  Versión/derivado orientado a formato IEEE (AI-native / 6G).
- **`framework_section.md`**  
  Sección enfocada al framework (UMRO-5G).
- **`taxonomy_comparative_section.md`**  
  Sección de taxonomía + análisis comparativo de literatura.

### Simulaciones
- **`simulations/`**: suite de simulaciones en Python.
  - **`run_all.py`**: ejecuta todas las simulaciones en secuencia y lista las figuras generadas.
  - **`figures/`**: salida con figuras `.png` generadas por los scripts.

---

## 2) Reproducibilidad: cómo ejecutar las simulaciones

### Requisitos
- Python 3.x
- Dependencias típicas:
  - `numpy`
  - `matplotlib`

> Nota: los scripts usan backend no interactivo (`Agg`) para generar figuras sin requerir entorno gráfico.

### Ejecutar toda la suite (recomendado)
Desde el directorio `5G_Orch/simulations`:

```bash
python3 run_all.py
```

Este comando:
1. Ejecuta las simulaciones en el orden definido.
2. Genera las figuras en `5G_Orch/simulations/figures/`.
3. Imprime un resumen (tiempos, estado OK/FAILED y listado de PNGs).

---

## 3) Simulaciones incluidas y figuras generadas

La suite se organiza como un “runner” unificado (`run_all.py`) con 5 simulaciones:

### Simulation 1 — Multi-Slice Resource Allocation (Monte Carlo)
- Script: `simulations/sim_multi_slice.py`
- Propósito: comparar **Hard Isolation**, **Soft Isolation** y una asignación dinámica tipo **UMRO-5G** en un escenario con 3 slices: `eMBB`, `URLLC`, `mMTC`.
- Métricas/Resultados principales:
  - Throughput vs carga
  - Utilización de PRBs vs carga
  - Probabilidad de violación de latencia (enfocada en URLLC)
- Figuras generadas:
  - `simulations/figures/fig1_throughput_vs_load.png`
  - `simulations/figures/fig2_utilization_vs_load.png`
  - `simulations/figures/fig3_latency_violation.png`

### Simulation 2 — DRL Convergence Comparison
- Script: `simulations/sim_drl_convergence.py`
- Propósito: generar curvas comparativas de convergencia de enfoques DRL (según el artículo).
- Figura:
  - `simulations/figures/fig4_drl_convergence.png`

### Simulation 3 — Scheduling Algorithm Comparison
- Script: `simulations/sim_scheduling.py`
- Propósito: comparación numérica de algoritmos de planificación (scheduling) y efectos en throughput/fairness/cobertura.
- Figuras:
  - `simulations/figures/fig5a_avg_throughput.png`
  - `simulations/figures/fig5b_fairness.png`
  - `simulations/figures/fig5c_p5_throughput.png`
  - `simulations/figures/fig5d_cell_edge.png`
  - `simulations/figures/fig5e_cdf_throughput.png`

### Simulation 4 — SFC Latency Sensitivity Analysis
- Script: `simulations/sim_sfc_latency.py`
- Propósito: análisis de sensibilidad de latencia en **Service Function Chaining (SFC)**.
- Figuras:
  - `simulations/figures/fig6_sfc_latency.png`
  - `simulations/figures/fig6b_sfc_latency_combined.png`

### Simulation 5 — Computational Complexity Comparison
- Script: `simulations/sim_complexity.py`
- Propósito: comparar complejidad computacional (tendencias/órdenes) de enfoques tratados.
- Figura:
  - `simulations/figures/fig7_complexity.png`

---

## 4) Notas de uso y estructura de salida

- Todas las figuras se almacenan en:
  - `5G_Orch/simulations/figures/`
- Si deseas limpiar resultados previos:
  - elimina manualmente los `.png` de `figures/` y vuelve a correr `run_all.py`.

---

## 5) Referencia rápida (árbol de archivos)
```

5G_Orch/
├─ Management_Orchestration_5G_Resources_MDPI_Computation.md
├─ Management_Orchestration_5G_AI_Native_6G_IEEE.md
├─ framework_section.md
├─ taxonomy_comparative_section.md
└─ simulations/
   ├─ run_all.py
   ├─ sim_multi_slice.py
   ├─ sim_drl_convergence.py
   ├─ sim_scheduling.py
   ├─ sim_sfc_latency.py
   ├─ sim_complexity.py
   └─ figures/
      ├─ fig1_throughput_vs_load.png
      ├─ fig2_utilization_vs_load.png
      ├─ fig3_latency_violation.png
      ├─ fig4_drl_convergence.png
      ├─ fig5a_avg_throughput.png
      ├─ fig5b_fairness.png
      ├─ fig5c_p5_throughput.png
      ├─ fig5d_cell_edge.png
      ├─ fig5e_cdf_throughput.png
      ├─ fig6_sfc_latency.png
      ├─ fig6b_sfc_latency_combined.png
      └─ fig7_complexity.png
```

---

## 6) Créditos / contexto académico

El manuscrito principal presenta un survey integral sobre **gestión de recursos y orquestación en 5G**, integrando fundamentos de RRM, NFV/SDN/MANO, slicing y control basado en ML/DRL, y proponiendo el framework **UMRO-5G** con bucles de control multi-timescale (Fast/Medium/Slow) y coordinación multi-dominio (RAN + Transporte + Core).

