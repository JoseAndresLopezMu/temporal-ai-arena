---
title: Temporal Arena Pro
emoji: 🎯
colorFrom: red
colorTo: purple
sdk: gradio
sdk_version: 5.31.0
app_file: app.py
pinned: true
license: apache-2.0
short_description: Pipeline completo forecasting — CRPS · Chronos · LightGBM
tags:
  - time-series
  - forecasting
  - probabilistic
  - chronos
  - gift-eval
  - crps
  - lightgbm
  - feature-engineering
  - gradio
---

# 🎯 Temporal Arena Pro — Pipeline Completo de Forecasting

La **cadena completa del dato** en forecasting: desde la ingesta hasta el pronóstico con intervalos de confianza.

```
📥 Ingesta → 🔬 EDA → ⚙️ Features → 🤖 Modelos → 📊 Evaluación → 🔮 Pronóstico
```

---

## 🌟 7 Etapas del Pipeline

### 📥 Etapa 1 — Ingesta de Datos
- **🌡️ OpenMeteo API** — Temperatura, precipitación, viento de cualquier ciudad (gratis)
- **📈 Yahoo Finance** — BTC, S&P 500, NVIDIA, Apple, ETH, Google
- **📊 GIFT-Eval datasets** — ETTh1 (Electricity Transformer Temperature)
- **📈 AirPassengers** — Clásico mensual (baseline estacional)
- **⚡ Consumo Energético** — Dataset sintético con doble estacionalidad
- **📁 CSV propio** — Sube tus datos (2 columnas: fecha, valor)

### 🔬 Etapa 2 — Análisis Exploratorio (EDA)
- ✅ Informe de calidad: outliers, faltantes, huecos, asimetría
- ✅ Tests de estacionariedad: **ADF + KPSS** con interpretación automática
- ✅ **Descomposición STL** robusta con métricas de fuerza tendencia/estacionalidad
- ✅ **ACF / PACF** con bandas de confianza y detección de patrones
- ✅ Distribución + **Q-Q plot** + Shapiro-Wilk normalidad
- ✅ **Periodograma** — detecta períodos dominantes via FFT + detrend

### ⚙️ Etapa 3 — Feature Engineering
- ✅ **Calendar features**: mes, día semana, trimestre, fin de semana
- ✅ **Lag features**: configurable (1 a 60 lags)
- ✅ **Rolling statistics**: media, std, min, max en múltiples ventanas
- ✅ **Fourier features**: sin/cos para capturar estacionalidad no lineal
- ✅ **Heatmap de correlación** features ↔ target

### 🤖 Etapa 4 — Modelos
| Modelo | Tipo | Incertidumbre |
|---|---|---|
| **Chronos-Bolt Small/Base** | Foundation IA (Amazon 2024) | 20-100 muestras Monte Carlo |
| **LightGBM** | ML + lag features | Bootstrap de residuos |
| **SARIMA** | Estadístico clásico | Simulación paramétrica |
| **ETS** (Holt-Winters) | Estadístico clásico | Bootstrap de residuos |
| **Seasonal Naïve** | Baseline | Ruido gaussiano |
| **Naïve** | Baseline | Ruido gaussiano |
| **Ensemble** | Combinación | Pool de muestras |

### 📊 Etapa 5 — Evaluación Probabilística
- 🏆 **Leaderboard automático** por CRPS (la métrica de GIFT-Eval y M4/M5)
- 📈 **Diagrama de calibración** (reliability diagram) — ¿cumples lo que prometes?
- 🎯 **Winkler Score** — calidad del intervalo de confianza 80%
- 📏 **Sharpness** — distribución del ancho de los ICs
- 🔬 **Análisis de residuos** — tiempo, distribución, Q-Q, vs. predicción
- ✅ **Test Ljung-Box** — ¿quedaron patrones sin capturar?
- 🔄 **Feature Importance** de LightGBM

### 🔮 Etapa 6 — Pronóstico Final
- Fan chart con 2 bandas de confianza (IC 50% + IC 80%) por modelo
- Tabla de predicciones: mediana + IC10% + IC90%
- **Exportar CSV** con todas las predicciones

### 📚 Etapa 7 — Aprende Todo
Guía completa integrada: métricas, modelos, CV, teoría probabilística, roadmap y referencias.

---

## 📐 Métricas implementadas

| Métrica | Fórmula | ¿Qué mide? |
|---|---|---|
| **CRPS** | `E[|X-y|] - ½E[|X-X'|]` | Distribución completa (↓ mejor) |
| **Winkler 80** | `width + penalización_misses` | Calidad IC 80% (↓ mejor) |
| **Calibración** | Cobertura empírica vs. nominal | ¿Es honesto el modelo? |
| **Sharpness** | `E[IC_80% width]` | Informatividad de ICs |
| **MAE** | `mean(\|y-ŷ\|)` | Error absoluto medio |
| **RMSE** | `sqrt(mean((y-ŷ)²))` | Error cuadrático (sensible a outliers) |
| **sMAPE%** | `200·\|y-ŷ\|/(|y|+|ŷ|)` | Error relativo simétrico |
| **BIAS** | `mean(ŷ-y)` | Sesgo sistemático |

---

## 🔄 Rolling Cross-Validation Correcta

```
Expanding window (NO random split):
  [──────────────────] → test [H]  → CRPS₁
  [────────────────────] → test [H] → CRPS₂
  [──────────────────────] → test [H] → CRPS₃
  CRPS_CV = mean(CRPS₁, CRPS₂, CRPS₃, ...)
```

---

## 📖 Contexto académico

**GIFT-Eval** (Salesforce 2024) — el benchmark de referencia:
- 28 datasets · 144K series · 177M puntos
- Métrica primaria: **CRPS**
- Ranking 2026: Timer-S1 > Chronos-2 > TimesFM-2.5 > Moirai-2

Esta app usa **ETTh1** (dataset oficial de GIFT-Eval) y la misma métrica primaria.

---

## 🔗 Referencias

- [Chronos paper](https://arxiv.org/abs/2403.07815) — Amazon 2024
- [GIFT-Eval paper](https://arxiv.org/abs/2410.10393) — Salesforce 2024
- [Timer-S1 paper](https://arxiv.org/abs/2603.04791) — THUML 2026 (SOTA)
- [FPP3 (Hyndman)](https://otexts.com/fpp3/) — Libro gratuito referencia
- [ETT Dataset](https://github.com/zhouhaoyi/ETDataset)

---

*por [JoseAndresLopez](https://huggingface.co/JoseAndresLopez)*
