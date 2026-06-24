"""
Temporal Arena Pro — Pipeline Completo de Forecasting
Toda la cadena: Ingesta → EDA → Features → Modelos → Evaluación → Pronóstico
"""

import warnings
warnings.filterwarnings("ignore")

import io, requests
from datetime import datetime, timedelta

import gradio as gr


def _df_to_md(df: "pd.DataFrame") -> str:
    """Convert DataFrame to markdown table without requiring tabulate."""
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep    = "|" + "|".join(["---"] * len(cols)) + "|"
    rows   = []
    for _, row in df.iterrows():
        rows.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join([header, sep] + rows)
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

# ─────────────────────────────────────────────────────────────
#  PALETA & CONSTANTES
# ─────────────────────────────────────────────────────────────
HIST_C = "#60A5FA"
TMPL   = "plotly_dark"

MODEL_COLORS = {
    "Chronos":        "#F87171",
    "ETS":            "#A78BFA",
    "ARIMA":          "#34D399",
    "LightGBM":       "#FBBF24",
    "Naïve":          "#94A3B8",
    "Seasonal Naïve": "#FB923C",
    "Ensemble":       "#F0ABFC",
}

BANDS = [
    (0.05, 0.95, "rgba(255,255,255,0.05)", "IC 90%"),
    (0.10, 0.90, "rgba(255,255,255,0.08)", "IC 80%"),
    (0.20, 0.80, "rgba(255,255,255,0.12)", "IC 60%"),
    (0.30, 0.70, "rgba(255,255,255,0.18)", "IC 40%"),
    (0.40, 0.60, "rgba(255,255,255,0.26)", "IC 20%"),
]

# ─────────────────────────────────────────────────────────────
#  ESTADO GLOBAL DE SESIÓN
# ─────────────────────────────────────────────────────────────
S: dict = {
    "df":        None,   # DataFrame crudo {fecha, valor}
    "source":    "",
    "feat_df":   None,   # DataFrame con features engineered
    "feat_cols": [],
    "period":    None,   # período estacional detectado
    "models":    {},     # {name: fitted model / dict}
    "forecasts": {},     # {name: {"median","q10","q90","samples"}}
    "metrics":   {},     # {name: {crps, winkler, mae, rmse, smape}}
    "horizon":   24,
    "split_idx": None,
}

# ─────────────────────────────────────────────────────────────
#  1. INGESTA DE DATOS
# ─────────────────────────────────────────────────────────────

_DS_CACHE: dict[str, pd.DataFrame] = {}

CIUDADES = {
    "Madrid":           (40.42, -3.70),
    "Barcelona":        (41.39,  2.17),
    "Londres":          (51.51, -0.13),
    "Nueva York":       (40.71,-74.01),
    "Tokio":            (35.68,139.69),
    "Buenos Aires":    (-34.60,-58.38),
}
METEO_VARS = {
    "🌡️ Temperatura máx (°C)":  "temperature_2m_max",
    "🌡️ Temperatura mín (°C)":  "temperature_2m_min",
    "🌧️ Precipitación (mm)":    "precipitation_sum",
    "💨 Viento máx (km/h)":     "wind_speed_10m_max",
    "☀️ Radiación solar (W/m²)": "shortwave_radiation_sum",
}
FINANCE_TICKERS = {
    "₿ Bitcoin (BTC-USD)":   "BTC-USD",
    "📈 S&P 500 (SPY)":       "SPY",
    "🤖 NVIDIA (NVDA)":       "NVDA",
    "🍎 Apple (AAPL)":        "AAPL",
    "Ξ Ethereum (ETH-USD)":  "ETH-USD",
    "🔍 Google (GOOGL)":      "GOOGL",
}


def fetch_weather(city: str, var_label: str, days: int) -> pd.DataFrame | str:
    lat, lon = CIUDADES[city]
    var = METEO_VARS[var_label]
    end   = datetime.today().date()
    start = end - timedelta(days=int(days))
    url = (f"https://archive-api.open-meteo.com/v1/archive"
           f"?latitude={lat}&longitude={lon}&start_date={start}&end_date={end}"
           f"&daily={var}&timezone=auto")
    try:
        r = requests.get(url, timeout=20); r.raise_for_status()
        d = r.json()
        return pd.DataFrame({"fecha": pd.to_datetime(d["daily"]["time"]),
                              "valor": d["daily"][var]}).dropna()
    except Exception as e:
        return f"Error OpenMeteo: {e}"


def fetch_finance(label: str, days: int) -> pd.DataFrame | str:
    try:
        import yfinance as yf
        ticker = FINANCE_TICKERS[label]
        end    = datetime.today().date()
        start  = end - timedelta(days=int(days))
        raw    = yf.download(ticker, start=str(start), end=str(end),
                             auto_adjust=True, progress=False)
        if raw.empty:
            return f"Sin datos para {ticker}"
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        df = raw[["Close"]].reset_index()
        df.columns = ["fecha", "valor"]
        return df.dropna()
    except Exception as e:
        return f"Error Yahoo Finance: {e}"


def fetch_etth1() -> pd.DataFrame | str:
    if "etth1" in _DS_CACHE:
        return _DS_CACHE["etth1"]
    url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
    try:
        raw = pd.read_csv(io.StringIO(requests.get(url, timeout=30).text))
        df = pd.DataFrame({"fecha": pd.to_datetime(raw["date"]),
                           "valor": raw["OT"].astype(float)})
        _DS_CACHE["etth1"] = df
        return df
    except Exception as e:
        return f"Error ETTh1: {e}"


def _airpassengers() -> pd.DataFrame:
    vals = [112,118,132,129,121,135,148,148,136,119,104,118,
            115,126,141,135,125,149,170,170,158,133,114,140,
            145,150,178,163,172,178,199,199,184,162,146,166,
            171,180,193,181,183,218,230,242,209,191,172,194,
            196,196,236,235,229,243,264,272,237,211,180,201,
            204,188,235,227,234,264,302,293,259,229,203,229,
            242,233,267,269,270,315,364,347,312,274,237,278,
            284,277,317,313,318,374,413,405,355,306,271,306,
            315,301,356,348,355,422,465,467,404,347,305,336,
            340,318,362,348,363,435,491,505,404,359,310,337,
            360,342,406,396,420,472,548,559,463,407,362,405,
            417,391,419,461,472,535,622,606,508,461,390,432]
    return pd.DataFrame({"fecha": pd.date_range("1949-01", periods=144, freq="ME"),
                         "valor": vals})


def _energia() -> pd.DataFrame:
    np.random.seed(2024); n=365*3; t=np.arange(n)
    vals = (200+0.08*t + 45*np.sin(2*np.pi*t/365-np.pi/2)
            + 20*np.sin(2*np.pi*t/7) + np.random.normal(0,8,n)).round(1)
    return pd.DataFrame({"fecha": pd.date_range("2022-01-01",periods=n,freq="D"), "valor":vals})


def parse_csv(f) -> pd.DataFrame | str:
    try:
        df = pd.read_csv(f).iloc[:,:2].copy()
        df.columns = ["fecha","valor"]
        df["fecha"] = pd.to_datetime(df["fecha"])
        df["valor"] = pd.to_numeric(df["valor"],errors="coerce")
        return df.dropna().sort_values("fecha").reset_index(drop=True)
    except Exception as e:
        return f"Error CSV: {e}"


def _store(df: pd.DataFrame, source: str):
    S["df"] = df.copy()
    S["source"] = source
    S["feat_df"] = None; S["feat_cols"] = []; S["period"] = None
    S["models"] = {}; S["forecasts"] = {}; S["metrics"] = {}


def _freq_days(df: pd.DataFrame) -> int:
    return max(1, int(df["fecha"].diff().dropna().dt.days.median()))


# ─────────────────────────────────────────────────────────────
#  2. DIAGNÓSTICOS EDA
# ─────────────────────────────────────────────────────────────

def detect_period(vals: np.ndarray, fd: int) -> int:
    """Detecta el período estacional dominante via periodograma con detrend."""
    from scipy.signal import periodogram, detrend as sp_detrend
    n = len(vals)
    if fd == 1:    candidates = [7, 14, 30, 365]
    elif fd <= 8:  candidates = [4, 13, 26, 52]
    elif fd <= 32: candidates = [4, 6, 12, 24]
    else:          candidates = [2, 3, 4, 7, 11]

    # Detrend lineal antes del análisis espectral (elimina falsa freq. DC)
    y = sp_detrend(vals.astype(float))
    f, Pxx = periodogram(y)
    Pxx[0] = 0

    with np.errstate(divide="ignore", invalid="ignore"):
        periods_arr = np.where(f > 0, 1.0 / f, np.inf)
    valid = (periods_arr >= 2) & (periods_arr <= n // 3)
    if not valid.any():
        return candidates[0]

    best_f = f[valid][np.argmax(Pxx[valid])]
    period = int(round(1.0 / best_f)) if best_f > 0 else candidates[0]

    # Snap al candidato más cercano (dentro del 35%)
    dists_rel = [abs(period - c) / max(period, 1) for c in candidates]
    if min(dists_rel) < 0.35:
        return candidates[int(np.argmin(dists_rel))]
    return int(np.clip(period, 2, n // 4))


def quality_report(df: pd.DataFrame) -> str:
    n       = len(df)
    missing = df["valor"].isna().sum()
    fd      = _freq_days(df)
    dups    = df["fecha"].duplicated().sum()
    expected_range = pd.date_range(df["fecha"].min(), df["fecha"].max(), freq=f"{fd}D")
    gaps    = len(expected_range) - n
    q1, q3  = df["valor"].quantile(0.25), df["valor"].quantile(0.75)
    iqr     = q3 - q1
    outliers = ((df["valor"] < q1 - 3*iqr) | (df["valor"] > q3 + 3*iqr)).sum()

    issues = []
    if missing > 0:  issues.append(f"⚠️ {missing} valores faltantes")
    if dups > 0:     issues.append(f"⚠️ {dups} fechas duplicadas")
    if gaps > 0:     issues.append(f"⚠️ ~{gaps} huecos temporales")
    if outliers > 0: issues.append(f"⚠️ {outliers} outliers (±3×IQR)")
    status = "✅ Sin problemas detectados" if not issues else "\n".join(issues)

    return f"""### 🔍 Informe de Calidad de Datos

| Métrica | Valor |
|---|---|
| Observaciones | **{n:,}** |
| Frecuencia estimada | **{fd}d** |
| Rango | **{df['fecha'].iloc[0].date()} → {df['fecha'].iloc[-1].date()}** |
| Días totales | **{(df['fecha'].iloc[-1]-df['fecha'].iloc[0]).days:,}** |
| Valores faltantes | **{missing}** |
| Fechas duplicadas | **{dups}** |
| Huecos estimados | **{gaps}** |
| Outliers (3×IQR) | **{outliers}** |
| Min / Max | **{df['valor'].min():.3f} / {df['valor'].max():.3f}** |
| Media / Mediana | **{df['valor'].mean():.3f} / {df['valor'].median():.3f}** |
| Desv. estándar | **{df['valor'].std():.3f}** |
| Coef. variación | **{df['valor'].std()/abs(df['valor'].mean())*100:.1f}%** |
| Asimetría | **{df['valor'].skew():.3f}** |
| Curtosis | **{df['valor'].kurt():.3f}** |

**Estado:** {status}
"""


def stationarity_report(vals: np.ndarray) -> str:
    from statsmodels.tsa.stattools import adfuller, kpss
    # ADF
    adf_r = adfuller(vals, autolag="AIC")
    adf_p = adf_r[1]
    adf_ok = adf_p < 0.05
    # KPSS
    try:
        kpss_r = kpss(vals, regression="c", nlags="auto")
        kpss_p = kpss_r[1]
        kpss_ok = kpss_p > 0.05
    except Exception:
        kpss_p, kpss_ok = np.nan, None

    if adf_ok and kpss_ok:
        verdict = "✅ **ESTACIONARIA** — lista para modelar directamente"
        rec     = "Puedes usar ARIMA(p,0,q) sin diferenciación."
    elif not adf_ok and not kpss_ok:
        verdict = "❌ **NO ESTACIONARIA** — requiere transformación"
        rec     = "Prueba diferenciación (d=1) o transformación logarítmica."
    else:
        verdict = "⚠️ **RESULTADO MIXTO** — revisar manualmente"
        rec     = "Analiza la gráfica y considera diferenciación estacional."

    # Diff test
    diff_adf = adfuller(np.diff(vals), autolag="AIC")[1]
    return f"""### 📐 Tests de Estacionariedad

| Test | Estadístico | p-valor | Resultado |
|---|---|---|---|
| **ADF** (H₀: raíz unitaria) | `{adf_r[0]:.4f}` | `{adf_p:.4f}` | {'✅ Estacionaria' if adf_ok else '❌ No estacionaria'} |
| **KPSS** (H₀: estacionaria) | — | `{kpss_p:.4f}` | {'✅ Estacionaria' if kpss_ok else '❌ No estacionaria'} |
| **ADF tras diff(1)** | — | `{diff_adf:.4f}` | {'✅ Estacionaria' if diff_adf<0.05 else '❌ Aún no'} |

**Veredicto:** {verdict}

**Recomendación:** {rec}

> ADF: p < 0.05 → rechaza raíz unitaria → estacionaria
> KPSS: p > 0.05 → no rechaza estacionariedad → estacionaria
"""


def ljung_box_report(residuals: np.ndarray, lags: int = 10) -> str:
    """Test de autocorrelación en residuos del modelo."""
    try:
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb = acorr_ljungbox(residuals, lags=lags, return_df=True)
        p_vals = lb["lb_pvalue"].values
        any_sig = any(p < 0.05 for p in p_vals)
        status = "⚠️ Autocorrelación en residuos (modelo incompleto)" if any_sig else "✅ Residuos sin autocorrelación significativa"
        rows = "\n".join(f"| Lag {i+1} | `{p:.4f}` | {'⚠️' if p<0.05 else '✅'} |"
                         for i, p in enumerate(p_vals))
        return f"""### 🔬 Ljung-Box Test (residuos del modelo)

{status}

| Lag | p-valor | |
|---|---|---|
{rows}

> p < 0.05 → aún quedan patrones no capturados por el modelo
"""
    except Exception as e:
        return f"> Ljung-Box no disponible: {e}"


# ─────────────────────────────────────────────────────────────
#  3. FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame, period: int,
                      n_lags: int, n_fourier: int) -> pd.DataFrame:
    """Genera features de calendario, lags, rolling y Fourier."""
    out = df.copy()
    out["t"] = np.arange(len(out))

    # — Calendario —
    out["mes"]       = out["fecha"].dt.month
    out["dia_semana"]= out["fecha"].dt.dayofweek
    out["trimestre"] = out["fecha"].dt.quarter
    out["es_fin_semana"] = (out["dia_semana"] >= 5).astype(int)

    # — Lags —
    lag_cols = []
    for lag in range(1, n_lags + 1):
        col = f"lag_{lag}"
        out[col] = out["valor"].shift(lag)
        lag_cols.append(col)

    # — Rolling stats —
    roll_cols = []
    for w in [period // 2, period, period * 2]:
        if w < 2 or w >= len(df):
            continue
        out[f"roll_mean_{w}"] = out["valor"].shift(1).rolling(w).mean()
        out[f"roll_std_{w}"]  = out["valor"].shift(1).rolling(w).std()
        out[f"roll_min_{w}"]  = out["valor"].shift(1).rolling(w).min()
        out[f"roll_max_{w}"]  = out["valor"].shift(1).rolling(w).max()
        roll_cols += [f"roll_mean_{w}", f"roll_std_{w}",
                      f"roll_min_{w}",  f"roll_max_{w}"]

    # — Fourier features (capturan estacionalidad no lineal) —
    fourier_cols = []
    t_arr = out["t"].values
    for k in range(1, n_fourier + 1):
        out[f"sin_{period}_{k}"] = np.sin(2 * np.pi * k * t_arr / period)
        out[f"cos_{period}_{k}"] = np.cos(2 * np.pi * k * t_arr / period)
        fourier_cols += [f"sin_{period}_{k}", f"cos_{period}_{k}"]

    feat_cols = (["mes","dia_semana","trimestre","es_fin_semana","t"]
                 + lag_cols + roll_cols + fourier_cols)
    out = out.dropna()
    return out, [c for c in feat_cols if c in out.columns]


# ─────────────────────────────────────────────────────────────
#  4. MODELOS
# ─────────────────────────────────────────────────────────────

_chronos_pipe = None

def _load_chronos(model_id: str):
    global _chronos_pipe
    from chronos import BaseChronosPipeline
    _chronos_pipe = BaseChronosPipeline.from_pretrained(
        model_id, device_map="cpu", torch_dtype=torch.float32)


def chronos_forecast(train_vals: np.ndarray, horizon: int,
                     model_id: str, num_samples: int = 50) -> np.ndarray | str:
    """Retorna (horizon, num_samples) o string de error."""
    global _chronos_pipe
    if _chronos_pipe is None:
        try:
            _load_chronos(model_id)
        except Exception as e:
            return f"Chronos: {e}"
    try:
        ctx = torch.tensor(train_vals[-512:], dtype=torch.float32).unsqueeze(0)
        out = _chronos_pipe.predict(ctx, prediction_length=horizon,
                                    num_samples=num_samples)
        s = out[0].numpy() if isinstance(out, tuple) else out[0].numpy()
        return s.T  # (horizon, samples)
    except Exception as e:
        return f"Chronos predict: {e}"


def ets_forecast(train_vals: np.ndarray, horizon: int,
                 period: int, n_bootstrap: int = 50) -> np.ndarray:
    """ETS con bootstrap de residuos → (horizon, samples)."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    n  = len(train_vals)
    use_season = (period >= 2) and (n >= period * 2)
    try:
        m = ExponentialSmoothing(
            train_vals, trend="add",
            seasonal="add" if use_season else None,
            seasonal_periods=period if use_season else None,
        ).fit(optimized=True)
        point  = m.forecast(horizon)
        resid  = train_vals - m.fittedvalues
        std    = resid.std()
        rng    = np.random.default_rng(42)
        samples = point[:, None] + rng.normal(0, std, (horizon, n_bootstrap))
        return samples
    except Exception:
        alpha, s_ = 0.3, float(train_vals[0])
        for v in train_vals[1:]:
            s_ = alpha*float(v) + (1-alpha)*s_
        base = np.full(horizon, s_)
        rng  = np.random.default_rng(42)
        return base[:, None] + rng.normal(0, float(np.std(train_vals))*0.15,
                                           (horizon, n_bootstrap))


def arima_forecast(train_vals: np.ndarray, horizon: int,
                   period: int, n_bootstrap: int = 50) -> np.ndarray:
    """SARIMA con bootstrap → (horizon, samples)."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    try:
        p_seas = min(1, period // 4)
        m = SARIMAX(train_vals, order=(2,1,2),
                    seasonal_order=(p_seas,1,0,period),
                    enforce_invertibility=False,
                    enforce_stationarity=False).fit(disp=False)
        point    = m.forecast(horizon)
        sim      = m.simulate(horizon, repetitions=n_bootstrap,
                              anchor="end", random_state=42)
        samples  = np.array(sim)
        if samples.ndim == 2 and samples.shape[0] == horizon:
            return samples
        return point[:, None] + np.random.default_rng(42).normal(
            0, train_vals.std()*0.08, (horizon, n_bootstrap))
    except Exception:
        from statsmodels.tsa.arima.model import ARIMA
        try:
            m2    = ARIMA(train_vals, order=(2,1,2)).fit()
            point = m2.forecast(horizon)
            rng   = np.random.default_rng(42)
            return point[:, None] + rng.normal(
                0, train_vals.std()*0.1, (horizon, n_bootstrap))
        except Exception:
            base = np.full(horizon, train_vals[-1])
            return base[:,None] + np.random.default_rng(42).normal(
                0, train_vals.std()*0.2, (horizon, n_bootstrap))


def lgbm_forecast(feat_df: pd.DataFrame, feat_cols: list[str],
                  train_vals: np.ndarray, horizon: int,
                  period: int, n_bootstrap: int = 50) -> np.ndarray:
    """LightGBM con lag features y forecast recursivo → (horizon, samples)."""
    import lightgbm as lgb
    target = "valor"
    train_feat = feat_df[feat_df["valor"].isin(train_vals[:len(feat_df)])].copy()
    # Use the full feat_df but only positions corresponding to train
    # Simpler: use all feat_df rows up to train size
    n_train = len(train_vals)
    # Map: feat_df has fewer rows (dropna removed some), align by index
    all_vals = feat_df["valor"].values
    available = min(len(feat_df), n_train)
    X_tr = feat_df[feat_cols].values[:available]
    y_tr = feat_df[target].values[:available]
    model = lgb.LGBMRegressor(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1,
    )
    model.fit(X_tr, y_tr)
    S["models"]["LightGBM"] = model
    S["models"]["LightGBM_feat_cols"] = feat_cols

    # Recursive forecast
    n_lags = sum(1 for c in feat_cols if c.startswith("lag_"))
    history = list(train_vals)

    def _make_row(history, t_idx):
        """Build one feature row from history."""
        row = {}
        row["t"] = float(t_idx)
        dt_last = feat_df["fecha"].iloc[-1] + pd.Timedelta(
            days=_freq_days(feat_df) * (t_idx - len(feat_df) + 1))
        row["mes"]          = float(dt_last.month)
        row["dia_semana"]   = float(dt_last.dayofweek)
        row["trimestre"]    = float(dt_last.quarter)
        row["es_fin_semana"]= float(dt_last.dayofweek >= 5)
        for lag in range(1, n_lags + 1):
            if lag <= len(history):
                row[f"lag_{lag}"] = float(history[-lag])
        # Rolling
        for col in feat_cols:
            if col.startswith("roll_") and col not in row:
                parts = col.split("_")
                w = int(parts[-1]) if parts[-1].isdigit() else int(parts[2])
                window = history[-w:] if len(history) >= w else history
                if "mean" in col: row[col] = float(np.mean(window))
                elif "std" in col: row[col] = float(np.std(window)) if len(window)>1 else 0.0
                elif "min" in col: row[col] = float(np.min(window))
                elif "max" in col: row[col] = float(np.max(window))
        # Fourier
        for col in feat_cols:
            if col.startswith("sin_") and col not in row:
                parts = col.split("_")
                per, k = int(parts[1]), int(parts[2])
                row[col] = float(np.sin(2*np.pi*k*t_idx/per))
            elif col.startswith("cos_") and col not in row:
                parts = col.split("_")
                per, k = int(parts[1]), int(parts[2])
                row[col] = float(np.cos(2*np.pi*k*t_idx/per))
        return [row.get(c, 0.0) for c in feat_cols]

    t0   = len(feat_df)
    point_preds = []
    h_copy = list(history)
    for h in range(horizon):
        row  = _make_row(h_copy, t0 + h)
        pred = float(model.predict([row])[0])
        point_preds.append(pred)
        h_copy.append(pred)

    point  = np.array(point_preds)
    resid_std = float(np.std(y_tr - model.predict(X_tr))) if len(X_tr) > 0 else train_vals.std()*0.1
    rng    = np.random.default_rng(42)
    grow   = np.sqrt(np.arange(1, horizon+1))
    noise  = rng.normal(0, resid_std, (horizon, n_bootstrap)) * grow[:, None] / np.sqrt(horizon)
    return point[:, None] + noise


def naive_samples(train_vals: np.ndarray, horizon: int,
                  n_bootstrap: int = 50) -> np.ndarray:
    std   = train_vals.std() * 0.15
    grow  = np.sqrt(np.arange(1, horizon+1))
    rng   = np.random.default_rng(42)
    base  = np.full(horizon, train_vals[-1])
    noise = rng.normal(0, std, (horizon, n_bootstrap)) * grow[:, None] / np.sqrt(horizon)
    return base[:, None] + noise


def seasonal_naive_samples(train_vals: np.ndarray, horizon: int,
                            period: int, n_bootstrap: int = 50) -> np.ndarray:
    tail  = train_vals[-period:]
    reps  = (horizon // period) + 2
    base  = np.tile(tail, reps)[:horizon]
    std   = train_vals.std() * 0.08
    rng   = np.random.default_rng(42)
    return base[:, None] + rng.normal(0, std, (horizon, n_bootstrap))


def ensemble_samples(forecasts: dict, weights: dict | None = None) -> np.ndarray:
    """Media ponderada de muestras de todos los modelos."""
    arrays = [v["samples"] for v in forecasts.values() if v.get("samples") is not None]
    if not arrays:
        return None
    min_h = min(a.shape[0] for a in arrays)
    arrays = [a[:min_h, :] for a in arrays]
    w      = np.ones(len(arrays)) / len(arrays)
    combined = sum(wi * a for wi, a in zip(w, arrays))
    return combined


# ─────────────────────────────────────────────────────────────
#  5. MÉTRICAS
# ─────────────────────────────────────────────────────────────

def crps_score(y_true: np.ndarray, samples: np.ndarray) -> float:
    N, M = samples.shape
    term1 = np.mean(np.abs(samples - y_true[:, None]), axis=1)
    s = np.sort(samples, axis=1)
    w = 2 * np.arange(1, M+1) - M - 1
    term2 = (s * w).sum(axis=1) / M**2
    return float(np.mean(term1 - term2))


def winkler_score(y_true: np.ndarray, samples: np.ndarray,
                  alpha: float = 0.80) -> float:
    lo = np.quantile(samples, (1-alpha)/2, axis=1)
    hi = np.quantile(samples, 1-(1-alpha)/2, axis=1)
    w  = hi - lo
    ml = 2/(1-alpha) * np.maximum(lo-y_true, 0)
    mr = 2/(1-alpha) * np.maximum(y_true-hi, 0)
    return float(np.mean(w + ml + mr))


def calibration_coverage(y_true: np.ndarray, samples: np.ndarray,
                          levels: np.ndarray) -> np.ndarray:
    return np.array([float(np.mean(y_true <= np.quantile(samples,q,axis=1)))
                     for q in levels])


def all_metrics(y_true: np.ndarray, samples: np.ndarray) -> dict:
    med  = np.median(samples, axis=1)
    mae  = float(np.mean(np.abs(y_true-med)))
    rmse = float(np.sqrt(np.mean((y_true-med)**2)))
    mask = y_true != 0
    mape = float(np.mean(np.abs((y_true[mask]-med[mask])/y_true[mask]))*100) if mask.any() else np.nan
    smape= float(np.mean(2*np.abs(y_true-med)/(np.abs(y_true)+np.abs(med)+1e-8))*100)
    bias = float(np.mean(med - y_true))
    crps = crps_score(y_true, samples)
    wink = winkler_score(y_true, samples)
    cov80 = float(np.mean(
        (y_true >= np.quantile(samples,0.10,axis=1)) &
        (y_true <= np.quantile(samples,0.90,axis=1))
    ) * 100)
    return {"MAE": round(mae,3), "RMSE": round(rmse,3),
            "MAPE%": round(mape,1), "sMAPE%": round(smape,1),
            "BIAS": round(bias,3),
            "CRPS": round(crps,4), "Winkler80": round(wink,4),
            "Cobertura80%": round(cov80,1)}


# ─────────────────────────────────────────────────────────────
#  6. VISUALIZACIONES
# ─────────────────────────────────────────────────────────────

def plot_series(df, period=None) -> go.Figure:
    roll = max(3, len(df)//20)
    ma   = df["valor"].rolling(roll, center=True).mean()
    fig  = go.Figure()
    fig.add_trace(go.Scatter(x=df["fecha"], y=df["valor"],
        mode="lines", name="Serie", line=dict(color=HIST_C, width=1.5),
        hovertemplate="%{x|%Y-%m-%d}: <b>%{y:.3f}</b><extra></extra>"))
    fig.add_trace(go.Scatter(x=df["fecha"], y=ma,
        mode="lines", name=f"MA({roll})", line=dict(color="#FBBF24",width=2,dash="dash")))
    if period:
        fig.add_annotation(x=0.01, y=0.97, xref="paper", yref="paper",
            text=f"Período detectado: <b>{period}</b>",
            showarrow=False, font=dict(color="#A78BFA",size=12),
            bgcolor="rgba(0,0,0,0.4)", borderpad=4)
    fig.update_layout(title="<b>Serie temporal + tendencia</b>", template=TMPL,
        height=380, margin=dict(t=50,b=30), hovermode="x unified",
        legend=dict(orientation="h",y=1.01))
    return fig


def _make_decomp_plot(df: pd.DataFrame, period: int) -> go.Figure | str:
    from statsmodels.tsa.seasonal import STL
    n = len(df)
    if n < period*2 + 5:
        return f"Necesitas al menos {period*2+5} puntos para descomponer."
    try:
        res = STL(df["valor"].values, period=period, robust=True).fit()
        fig = make_subplots(4,1, shared_xaxes=True, vertical_spacing=0.04,
            subplot_titles=("Original","Tendencia","Estacionalidad","Residuo"))
        for i,(s,c) in enumerate([
            (res.observed,"#60A5FA"),(res.trend,"#F87171"),
            (res.seasonal,"#A78BFA"),(res.resid,"#FBBF24")],1):
            fig.add_trace(go.Scatter(x=df["fecha"],y=s,mode="lines",
                line=dict(color=c,width=1.5),showlegend=False),row=i,col=1)
            if i==4: fig.add_hline(y=0,line_dash="dot",line_color="gray",
                                   opacity=0.5,row=i,col=1)
        Fs = max(0,1-np.var(res.resid)/(np.var(res.seasonal)+np.var(res.resid)))
        Ft = max(0,1-np.var(res.resid)/(np.var(res.trend  )+np.var(res.resid)))
        fig.add_annotation(x=0.01,y=0.99,xref="paper",yref="paper",
            text=f"Fuerza tendencia: <b>{Ft:.2f}</b> | Fuerza estacionalidad: <b>{Fs:.2f}</b>",
            showarrow=False,font=dict(size=11,color="#94A3B8"),
            bgcolor="rgba(0,0,0,0.5)",borderpad=4)
        fig.update_layout(title="<b>Descomposición STL robusta</b>",
            template=TMPL,height=700,margin=dict(t=60,b=20))
        return fig
    except Exception as e:
        return f"Error STL: {e}"


def plot_acf_pacf(vals: np.ndarray, lags: int = 40) -> go.Figure | str:
    from statsmodels.tsa.stattools import acf, pacf
    lags = min(lags, len(vals)//2-1)
    acf_v,_ = acf(vals,nlags=lags,alpha=0.05)
    pacf_v,_= pacf(vals,nlags=lags,alpha=0.05)
    ci = 1.96/np.sqrt(len(vals))
    fig = make_subplots(2,1,subplot_titles=("ACF — autocorrelación total",
        "PACF — autocorrelación parcial"),vertical_spacing=0.12)
    for r,(vr,lab) in enumerate([(acf_v,"ACF"),(pacf_v,"PACF")],1):
        for i,v in enumerate(vr):
            col = "#F87171" if abs(v)>ci else "#60A5FA"
            fig.add_trace(go.Bar(x=[i],y=[v],marker_color=col,
                showlegend=False,width=0.4),row=r,col=1)
        for s in [1,-1]:
            fig.add_hline(y=s*ci,line_dash="dash",line_color="white",
                opacity=0.4,row=r,col=1)
        fig.add_hline(y=0,line_color="gray",opacity=0.5,row=r,col=1)
    fig.update_layout(title="<b>ACF / PACF</b>  (barras rojas = significativas)",
        template=TMPL,height=520,margin=dict(t=60,b=30))
    return fig


def plot_distribution(vals: np.ndarray) -> go.Figure:
    from scipy import stats
    fig = make_subplots(1,2,subplot_titles=("Distribución (histograma+KDE)","Q-Q Plot (normalidad)"))
    # Histograma
    fig.add_trace(go.Histogram(x=vals,nbinsx=30,name="Distribución",
        marker_color="#60A5FA",opacity=0.7,histnorm="probability density"),row=1,col=1)
    # KDE
    kde_x = np.linspace(vals.min(), vals.max(), 200)
    kde_y = stats.gaussian_kde(vals)(kde_x)
    fig.add_trace(go.Scatter(x=kde_x,y=kde_y,mode="lines",name="KDE",
        line=dict(color="#F87171",width=2)),row=1,col=1)
    # Q-Q
    (osm, osr), (slope, intercept, r) = stats.probplot(vals, dist="norm")
    fig.add_trace(go.Scatter(x=osm,y=osr,mode="markers",name="Q-Q",
        marker=dict(color="#A78BFA",size=4)),row=1,col=2)
    line_x = np.array([osm[0],osm[-1]])
    fig.add_trace(go.Scatter(x=line_x,y=slope*line_x+intercept,mode="lines",
        name="Normal",line=dict(color="#F87171",width=2,dash="dash")),row=1,col=2)
    sw_stat, sw_p = stats.shapiro(vals[:5000])
    fig.add_annotation(x=0.75,y=0.05,xref="paper",yref="paper",
        text=f"Shapiro-Wilk p={sw_p:.4f}<br>{'Normal ✅' if sw_p>0.05 else 'No normal ⚠️'}",
        showarrow=False,font=dict(size=11,color="#94A3B8"),
        bgcolor="rgba(0,0,0,0.5)",borderpad=4)
    fig.update_layout(title="<b>Distribución y normalidad</b>",
        template=TMPL,height=420,margin=dict(t=60,b=30),showlegend=False)
    return fig


def plot_periodogram(vals: np.ndarray, fd: int) -> go.Figure:
    """Periodograma para detectar períodos dominantes."""
    from scipy.signal import periodogram
    f, Pxx = periodogram(vals - vals.mean())
    Pxx[0] = 0
    # Top 5 picos
    peaks_idx = np.argsort(Pxx)[-8:][::-1]
    periods_dom = []
    for idx in peaks_idx:
        if f[idx] > 0:
            periods_dom.append((1/f[idx], Pxx[idx]))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=1/f[1:],y=Pxx[1:],mode="lines",
        name="Potencia",line=dict(color="#60A5FA",width=1.5)))
    for per, pow_ in periods_dom[:3]:
        fig.add_vline(x=per,line_dash="dot",line_color="#F87171",opacity=0.7,
            annotation_text=f"T={per:.1f}",annotation_position="top")
    fig.update_layout(
        title="<b>Periodograma</b> — detecta períodos dominantes",
        xaxis=dict(title="Período",type="log",range=[0,3]),
        yaxis_title="Potencia espectral",
        template=TMPL,height=360,margin=dict(t=60,b=40))
    return fig


def plot_features_heatmap(feat_df: pd.DataFrame, feat_cols: list[str]) -> go.Figure:
    """Correlación de features con el target."""
    corrs = feat_df[feat_cols + ["valor"]].corr()["valor"].drop("valor")
    corrs = corrs.sort_values(ascending=False)
    top   = corrs.head(20)
    colors = ["#F87171" if v > 0 else "#60A5FA" for v in top.values]
    fig = go.Figure(go.Bar(x=top.index.tolist(), y=top.values,
        marker_color=colors, text=top.values.round(3), textposition="outside"))
    fig.update_layout(title="<b>Correlación features ↔ target</b> (top 20)",
        xaxis_tickangle=-45, template=TMPL,
        height=400, margin=dict(t=60,b=120))
    return fig


def plot_fan(df_train: pd.DataFrame, samples_dict: dict,
             test_vals=None, test_dates=None, horizon: int = 24) -> go.Figure:
    fd    = _freq_days(df_train)
    last  = df_train["fecha"].iloc[-1]
    fdates= pd.date_range(start=last+pd.Timedelta(days=fd),
                          periods=horizon, freq=f"{fd}D")
    show  = min(len(df_train), horizon*5)
    hist  = df_train.iloc[-show:]
    fig   = go.Figure()
    fig.add_trace(go.Scatter(x=hist["fecha"],y=hist["valor"],
        mode="lines",name="Histórico",line=dict(color=HIST_C,width=2),
        hovertemplate="%{x|%Y-%m-%d}: <b>%{y:.3f}</b><extra></extra>"))

    for name, samples in samples_dict.items():
        color = MODEL_COLORS.get(name,"#fff")
        lo80  = np.quantile(samples,0.10,axis=1)
        hi80  = np.quantile(samples,0.90,axis=1)
        lo50  = np.quantile(samples,0.25,axis=1)
        hi50  = np.quantile(samples,0.75,axis=1)
        med   = np.median(samples,axis=1)
        rgba  = color.replace("#","")
        r,g,b = int(rgba[0:2],16),int(rgba[2:4],16),int(rgba[4:6],16)
        fig.add_trace(go.Scatter(
            x=list(fdates)+list(fdates[::-1]),
            y=list(hi80)+list(lo80[::-1]),
            fill="toself",fillcolor=f"rgba({r},{g},{b},0.10)",
            line=dict(color="rgba(0,0,0,0)"),name=f"IC80% {name}",showlegend=True))
        fig.add_trace(go.Scatter(
            x=list(fdates)+list(fdates[::-1]),
            y=list(hi50)+list(lo50[::-1]),
            fill="toself",fillcolor=f"rgba({r},{g},{b},0.18)",
            line=dict(color="rgba(0,0,0,0)"),name=f"IC50% {name}",showlegend=True))
        fig.add_trace(go.Scatter(x=fdates,y=med,mode="lines",name=name,
            line=dict(color=color,width=2.5),
            hovertemplate=f"{name}: %{{y:.3f}}<extra></extra>"))

    if test_vals is not None:
        fig.add_trace(go.Scatter(x=test_dates,y=test_vals,mode="lines+markers",
            name="✅ Real",line=dict(color="white",width=2,dash="dot"),
            marker=dict(size=5,symbol="circle-open")))

    fig.add_vline(x=str(last),line_dash="dot",line_color="gray",opacity=0.5,
                  annotation_text="→",annotation_position="top right")
    fig.update_layout(title="<b>Fan Chart — Pronóstico multi-modelo</b>",
        template=TMPL,height=520,margin=dict(t=70,b=40),
        hovermode="x unified",
        legend=dict(orientation="h",y=1.01,font=dict(size=10)))
    return fig


def plot_calibration(y_true: np.ndarray, samples_dict: dict) -> go.Figure:
    levels = np.linspace(0.05,0.95,19)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=levels,y=levels,mode="lines",
        name="Perfecto",line=dict(color="gray",width=2,dash="dash")))
    # Zona de tolerancia ±5%
    fig.add_trace(go.Scatter(
        x=list(levels)+list(levels[::-1]),
        y=list(levels+0.05)+list((levels-0.05)[::-1]),
        fill="toself",fillcolor="rgba(255,255,255,0.05)",
        line=dict(color="rgba(0,0,0,0)"),name="Zona ±5%"))
    for name, samples in samples_dict.items():
        cov = calibration_coverage(y_true, samples, levels)
        fig.add_trace(go.Scatter(x=levels,y=cov,mode="lines+markers",
            name=name,line=dict(color=MODEL_COLORS.get(name,"#fff"),width=2),
            marker=dict(size=7),
            hovertemplate=f"Nominal: %{{x:.0%}}<br>Real: %{{y:.0%}}<extra>{name}</extra>"))
    fig.update_layout(
        title="<b>Diagrama de Calibración</b><br><sub>Diagonal = calibración perfecta</sub>",
        xaxis=dict(title="Nivel nominal",tickformat=".0%"),
        yaxis=dict(title="Cobertura real",tickformat=".0%"),
        template=TMPL,height=430,margin=dict(t=70,b=50),
        legend=dict(x=0.02,y=0.98))
    return fig


def plot_residuals(y_true: np.ndarray, forecasts: dict,
                   dates=None) -> go.Figure:
    fig = make_subplots(2,2,
        subplot_titles=("Residuos en el tiempo","Distribución de residuos",
                        "Residuos vs Predicción","Q-Q residuos"),
        vertical_spacing=0.12,horizontal_spacing=0.1)
    from scipy import stats as scipy_stats
    for name, fc in forecasts.items():
        color = MODEL_COLORS.get(name,"#fff")
        resid = y_true - fc["median"]
        x_ax  = list(range(len(resid))) if dates is None else dates
        fig.add_trace(go.Scatter(x=x_ax,y=resid,mode="lines",name=name,
            line=dict(color=color,width=1.5),showlegend=True),row=1,col=1)
        fig.add_trace(go.Histogram(x=resid,name=name,marker_color=color,
            opacity=0.6,histnorm="probability density",showlegend=False),row=1,col=2)
        fig.add_trace(go.Scatter(x=fc["median"],y=resid,mode="markers",name=name,
            marker=dict(color=color,size=5,opacity=0.7),showlegend=False),row=2,col=1)
        osm,osr = scipy_stats.probplot(resid,dist="norm")[0]
        fig.add_trace(go.Scatter(x=osm,y=osr,mode="markers",name=name,
            marker=dict(color=color,size=4),showlegend=False),row=2,col=2)
    fig.add_hline(y=0,line_dash="dot",line_color="gray",opacity=0.5,row=1,col=1)
    fig.add_hline(y=0,line_dash="dot",line_color="gray",opacity=0.5,row=2,col=1)
    fig.update_layout(title="<b>Análisis de Residuos</b>",
        template=TMPL,height=640,margin=dict(t=70,b=40))
    return fig


def plot_sharpness(y_true: np.ndarray, samples_dict: dict) -> go.Figure:
    fig = make_subplots(1,2,
        subplot_titles=("Ancho IC 80% por modelo","Cobertura real por nivel IC"))
    levels = [0.50,0.60,0.70,0.80,0.90]
    for name, samples in samples_dict.items():
        color = MODEL_COLORS.get(name,"#fff")
        lo,hi = np.quantile(samples,0.10,axis=1),np.quantile(samples,0.90,axis=1)
        fig.add_trace(go.Histogram(x=hi-lo,name=name,marker_color=color,
            opacity=0.65,nbinsx=20,histnorm="probability density"),row=1,col=1)
        coverages = [float(np.mean((y_true>=np.quantile(samples,(1-l)/2,axis=1))&
                                   (y_true<=np.quantile(samples,1-(1-l)/2,axis=1))))*100
                     for l in levels]
        fig.add_trace(go.Scatter(x=[f"{int(l*100)}%" for l in levels],
            y=coverages,mode="lines+markers",name=name,showlegend=False,
            line=dict(color=color,width=2)),row=1,col=2)
    # Línea ideal en col 2
    fig.add_trace(go.Scatter(x=[f"{int(l*100)}%" for l in levels],
        y=[l*100 for l in levels],mode="lines",name="Ideal",showlegend=False,
        line=dict(color="gray",dash="dash")),row=1,col=2)
    fig.update_layout(title="<b>Sharpness & Cobertura</b>",
        template=TMPL,height=400,margin=dict(t=60,b=40),barmode="overlay")
    return fig


def plot_rolling_crps(results: dict) -> go.Figure:
    fig = go.Figure()
    for name, vals in results.items():
        clean = [v for v in vals if not np.isnan(v)]
        if not clean: continue
        fig.add_trace(go.Scatter(
            x=list(range(1,len(vals)+1)), y=vals, mode="lines+markers",
            name=name, line=dict(color=MODEL_COLORS.get(name,"#fff"),width=2),
            marker=dict(size=7),
            hovertemplate=f"Ventana %{{x}}: CRPS=%{{y:.4f}}<extra>{name}</extra>"))
    fig.update_layout(title="<b>CRPS por ventana — Rolling Cross-Validation</b>",
        xaxis_title="Ventana",yaxis_title="CRPS (↓ mejor)",
        template=TMPL,height=400,margin=dict(t=60,b=50),
        legend=dict(orientation="h",y=1.01))
    return fig


def plot_feature_importance(model, feat_cols: list[str]) -> go.Figure | None:
    try:
        import lightgbm as lgb
        imp = pd.Series(model.feature_importances_, index=feat_cols)
        imp = imp.sort_values(ascending=False).head(20)
        fig = go.Figure(go.Bar(x=imp.index.tolist(), y=imp.values,
            marker_color="#FBBF24", text=imp.values, textposition="outside"))
        fig.update_layout(title="<b>Feature Importance — LightGBM</b>",
            xaxis_tickangle=-45, template=TMPL,
            height=420, margin=dict(t=60,b=140))
        return fig
    except Exception:
        return None


def leaderboard_md(metrics: dict) -> str:
    if not metrics:
        return "> Ejecuta el benchmark para ver el leaderboard."
    rows = sorted(metrics.items(), key=lambda x: x[1].get("CRPS", 999))
    medals = ["🥇","🥈","🥉","4️⃣","5️⃣","6️⃣","7️⃣"]
    md  = "### 🏆 Leaderboard\n\n"
    md += "| # | Modelo | CRPS ↓ | Winkler80 ↓ | sMAPE% ↓ | MAE ↓ | RMSE ↓ | BIAS | Cob80% |\n"
    md += "|---|---|---|---|---|---|---|---|---|\n"
    for i,(name,m) in enumerate(rows):
        med = medals[i] if i < len(medals) else ""
        md += (f"| {med} | **{name}** | `{m.get('CRPS','—')}` | `{m.get('Winkler80','—')}` "
               f"| `{m.get('sMAPE%','—')}` | `{m.get('MAE','—')}` | `{m.get('RMSE','—')}` "
               f"| `{m.get('BIAS','—')}` | `{m.get('Cobertura80%','—')}%` |\n")
    md += ("\n> **CRPS**: calidad distribución completa (↓ mejor) | "
           "**Winkler**: calidad IC 80% (↓ mejor) | "
           "**Cob80%**: cobertura real del IC 80% (ideal ≈ 80)")
    return md


# ─────────────────────────────────────────────────────────────
#  CALLBACKS
# ─────────────────────────────────────────────────────────────

def cb_load_weather(city, var, days):
    res = fetch_weather(city, var, days)
    if isinstance(res,str): return None,"❌ "+res,None,None,None,None
    _store(res, f"{var} · {city}")
    return _eda_outputs(res)

def cb_load_finance(label, days):
    res = fetch_finance(label, days)
    if isinstance(res,str): return None,"❌ "+res,None,None,None,None
    _store(res, label)
    return _eda_outputs(res)

def cb_load_etth1():
    res = fetch_etth1()
    if isinstance(res,str): return None,"❌ "+res,None,None,None,None
    _store(res, "ETTh1 — Oil Temperature")
    return _eda_outputs(res)

def cb_load_ap():
    res = _airpassengers()
    _store(res, "AirPassengers"); return _eda_outputs(res)

def cb_load_energia():
    res = _energia()
    _store(res, "Consumo Energético"); return _eda_outputs(res)

def cb_load_csv(file_obj):
    if file_obj is None: return None,"Sube un CSV primero.",None,None,None,None
    res = parse_csv(file_obj)
    if isinstance(res,str): return None,"❌ "+res,None,None,None,None
    _store(res, "CSV propio"); return _eda_outputs(res)

def _eda_outputs(df):
    fd     = _freq_days(df)
    period = detect_period(df["valor"].values, fd)
    S["period"] = period
    fig_s  = plot_series(df, period)
    qr_md  = quality_report(df)
    stat_md= stationarity_report(df["valor"].values)
    return fig_s, qr_md, stat_md, None, None, None

def cb_run_eda():
    df = S.get("df")
    if df is None: return None,None,None,None
    period = S.get("period",12)
    fd = _freq_days(df)
    fig_d   = _make_decomp_plot(df, period)
    fig_ap  = plot_acf_pacf(df["valor"].values)
    fig_di  = plot_distribution(df["valor"].values)
    fig_pg  = plot_periodogram(df["valor"].values, fd)
    return (fig_d if not isinstance(fig_d,str) else None,
            fig_ap if not isinstance(fig_ap,str) else None,
            fig_di, fig_pg)

def cb_engineer(n_lags, n_fourier):
    df = S.get("df")
    if df is None: return None,"⚠️ Carga datos primero.",None
    period = S.get("period",12)
    feat_df, feat_cols = engineer_features(df, period, int(n_lags), int(n_fourier))
    S["feat_df"]   = feat_df
    S["feat_cols"] = feat_cols
    fig_h = plot_features_heatmap(feat_df, feat_cols)
    preview = _df_to_md(feat_df[["fecha","valor"]+feat_cols[:8]].head(8))
    md  = f"### ✅ Features generadas: {len(feat_cols)} columnas\n\n"
    md += f"**Período estacional:** {period} | **Lags:** {n_lags} | **Fourier términos:** {n_fourier}\n\n"
    md += "**Features:**\n" + ", ".join(f"`{c}`" for c in feat_cols[:30])
    if len(feat_cols) > 30: md += f", ... (+{len(feat_cols)-30} más)"
    md += f"\n\n**Primeras filas:**\n\n{preview}"
    return fig_h, md, None

def cb_run_models(horizon, model_id, num_samples, use_chronos,
                  use_ets, use_arima, use_lgbm, use_naive, use_snaive):
    df = S.get("df")
    if df is None: return [None]*8 + ["⚠️ Carga datos primero."]
    horizon = int(horizon); num_samples = int(num_samples)
    n = len(df); period = S.get("period",12)
    split = max(horizon, n//5)
    S["horizon"]   = horizon
    S["split_idx"] = n - split
    train_df = df.iloc[:n-split]
    test_df  = df.iloc[n-split:n-split+horizon]
    train_v  = train_df["valor"].values.astype(float)
    test_v   = test_df["valor"].values.astype(float)[:horizon]
    test_dates = test_df["fecha"].values[:horizon]

    forecasts = {}
    metrics   = {}

    def _register(name, samples_or_err):
        if isinstance(samples_or_err, str):
            return
        s = samples_or_err
        med  = np.median(s,axis=1)
        q10  = np.quantile(s,0.10,axis=1)
        q90  = np.quantile(s,0.90,axis=1)
        forecasts[name] = {"samples":s,"median":med,"q10":q10,"q90":q90}
        metrics[name]   = all_metrics(test_v[:len(med)], s)

    if use_chronos:
        _register("Chronos", chronos_forecast(train_v, horizon, model_id, num_samples))
    if use_ets:
        _register("ETS", ets_forecast(train_v, horizon, period, num_samples))
    if use_arima:
        _register("ARIMA", arima_forecast(train_v, horizon, period, num_samples))
    if use_lgbm and S.get("feat_df") is not None:
        _register("LightGBM",
                  lgbm_forecast(S["feat_df"].iloc[:n-split], S["feat_cols"],
                                train_v, horizon, period, num_samples))
    if use_naive:
        _register("Naïve", naive_samples(train_v, horizon, num_samples))
    if use_snaive:
        _register("Seasonal Naïve", seasonal_naive_samples(train_v, horizon, period, num_samples))

    # Ensemble
    if len(forecasts) >= 2:
        ens_s = ensemble_samples(forecasts)
        if ens_s is not None:
            _register("Ensemble", ens_s)

    S["forecasts"] = forecasts
    S["metrics"]   = metrics

    if not forecasts:
        return [None]*8 + ["⚠️ Ningún modelo produjo resultados."]

    samples_dict = {k: v["samples"] for k,v in forecasts.items()}
    fc_dict_full = {k:v for k,v in forecasts.items()}

    fig_fan_  = plot_fan(train_df, samples_dict, test_v, test_dates, horizon)
    fig_cal   = plot_calibration(test_v, samples_dict)
    fig_resid_= plot_residuals(test_v, fc_dict_full, test_dates)
    fig_sharp = plot_sharpness(test_v, samples_dict)

    # Feature importance para LightGBM
    fig_fi = None
    if "LightGBM" in S.get("models",{}):
        fig_fi = plot_feature_importance(S["models"]["LightGBM"], S["feat_cols"])

    board_md = leaderboard_md(metrics)

    # Residuos del mejor modelo (Ljung-Box)
    best = min(metrics, key=lambda x: metrics[x]["CRPS"])
    resid_series = test_v - forecasts[best]["median"][:len(test_v)]
    lb_md = ljung_box_report(resid_series)

    return (fig_fan_, fig_cal, fig_resid_, fig_sharp,
            fig_fi, board_md, lb_md, None, None)

def cb_future_forecast(horizon):
    import tempfile, os
    df = S.get("df")
    if df is None: return None, "⚠️ Carga datos primero.", gr.update(visible=False)
    horizon  = int(horizon)
    period   = S.get("period",12)
    num_s    = 50
    train_v  = df["valor"].values.astype(float)
    forecasts= {}

    def _reg(name, s):
        if not isinstance(s, str):
            forecasts[name] = s

    # Re-usar forecasts existing o regenerar
    existing = S.get("forecasts",{})
    if existing:
        for name in existing:
            fc = S["forecasts"][name]
            _reg(name, fc["samples"])
    else:
        _reg("ETS",   ets_forecast(train_v, horizon, period, num_s))
        _reg("Naïve", naive_samples(train_v, horizon, num_s))

    fig = plot_fan(df, forecasts, horizon=horizon)

    # Tabla de predicciones
    fd    = _freq_days(df)
    last  = df["fecha"].iloc[-1]
    fdates= pd.date_range(start=last+pd.Timedelta(days=fd), periods=horizon, freq=f"{fd}D")
    rows  = []
    for name, s in forecasts.items():
        med = np.median(s,axis=1)[:horizon]
        lo  = np.quantile(s,0.10,axis=1)[:horizon]
        hi  = np.quantile(s,0.90,axis=1)[:horizon]
        for i,(d,m,l,h) in enumerate(zip(fdates,med,lo,hi)):
            rows.append({"Fecha":d.date(),"Modelo":name,
                         "Mediana":round(float(m),3),
                         "IC10%":round(float(l),3),"IC90%":round(float(h),3)})
    df_table = pd.DataFrame(rows)
    table_preview = _df_to_md(df_table.head(20))
    md = f"### Pronóstico futuro — {horizon} pasos\n\n{table_preview}"
    if len(df_table)>20: md += f"\n\n*...y {len(df_table)-20} filas más (descarga el CSV)*"

    # Escribir CSV a fichero temporal (gr.DownloadButton necesita path, no bytes)
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, encoding="utf-8")
    df_table.to_csv(tmp, index=False)
    tmp.close()

    return fig, md, gr.update(visible=True, value=tmp.name)

def cb_rolling_cv(horizon, n_win, model_id):
    df = S.get("df")
    if df is None: return None,"⚠️ Carga datos primero."
    vals   = df["valor"].values.astype(float)
    period = S.get("period",12)
    n      = len(vals)
    horizon= int(horizon); n_win=int(n_win)
    min_tr = max(period*3, n//3)
    step   = max(horizon, (n-min_tr-horizon)//n_win)
    windows= []
    t = min_tr
    while t+horizon<=n and len(windows)<n_win:
        windows.append((t, t+horizon)); t+=step
    if not windows:
        return None, "No hay suficientes datos para CV."

    results = {"ETS":[],"Naïve":[],"Seasonal Naïve":[]}
    if model_id:
        results["Chronos"] = []

    for tr_end, te_end in windows:
        tr = vals[:tr_end]; te = vals[tr_end:te_end]
        ets_s  = ets_forecast(tr, horizon, period, 20)
        nai_s  = naive_samples(tr, horizon, 20)
        snai_s = seasonal_naive_samples(tr, horizon, period, 20)
        results["ETS"].append(crps_score(te, ets_s))
        results["Naïve"].append(crps_score(te, nai_s))
        results["Seasonal Naïve"].append(crps_score(te, snai_s))
        if "Chronos" in results:
            s = chronos_forecast(tr, horizon, model_id, 20)
            results["Chronos"].append(crps_score(te,s) if not isinstance(s,str) else np.nan)

    fig = plot_rolling_crps(results)
    md  = "### CRPS medio (Rolling CV)\n\n| Modelo | CRPS medio |\n|---|---|\n"
    for name,vals_r in sorted(results.items(), key=lambda x: np.nanmean(x[1])):
        md += f"| **{name}** | `{np.nanmean(vals_r):.4f}` |\n"
    return fig, md

# ─────────────────────────────────────────────────────────────
#  CONTENIDO EDUCATIVO
# ─────────────────────────────────────────────────────────────

LEARN_MD = """
# 📚 Guía Completa de Forecasting — De 0 a Producción

---

## ETAPA 1 — Ingesta y Calidad de Datos

La calidad del dato es el 80% del trabajo real. Antes de modelar:

| Check | Qué buscar | Solución |
|---|---|---|
| **Valores faltantes** | NaN en la serie | Interpolación lineal, forward-fill |
| **Outliers** | Picos imposibles | Winsorización, imputación por IQR |
| **Duplicados** | Fechas repetidas | Aggregar o eliminar |
| **Frecuencia irregular** | Saltos temporales | Resamplear a freq. fija |
| **Escala** | Valores muy grandes | Normalización (min-max o z-score) |
| **No negativos** | ¿Ventas < 0? | Transformación logarítmica |

---

## ETAPA 2 — Análisis Exploratorio (EDA)

### Descomposición STL
`y(t) = Tendencia(t) + Estacionalidad(t) + Residuo(t)`

- **Tendencia**: dirección a largo plazo (¿crece? ¿decrece?)
- **Estacionalidad**: patrón periódico (semanas, meses, años)
- **Residuo**: lo que no se explica → mide el "ruido" real

**Fuerza de estacionalidad** Fs ∈ [0,1]:
```
Fs = max(0, 1 - Var(residuo) / Var(residuo + estacionalidad))
```
Fs > 0.64 → estacionalidad pronunciada (la M4 competition usaba este umbral)

### ACF y PACF
- **ACF(k)**: correlación entre xₜ y xₜ₋ₖ
  - Cae lentamente → no estacionaria
  - Picos en k=12,24... → estacionalidad de período 12
- **PACF(k)**: correlación directa (eliminando intermediarios)
  - Caída brusca en lag=p → modelo AR(p)

### Tests de Estacionariedad
- **ADF** (H₀: raíz unitaria): p < 0.05 → estacionaria
- **KPSS** (H₀: estacionaria): p > 0.05 → estacionaria
- Ambos juntos dan más certeza

### Periodograma
Análisis espectral via FFT: muestra qué frecuencias dominan.
Pico en período=52 → ciclo anual (datos semanales).

### Test de normalidad (Shapiro-Wilk)
- p > 0.05 → no rechazamos normalidad
- Importante para: intervalos de predicción gaussianos, tests paramétricos

---

## ETAPA 3 — Feature Engineering

### Calendar Features
`mes`, `dia_semana`, `trimestre`, `es_fin_semana`
→ Capturan estacionalidad conocida sin que el modelo la aprenda

### Lag Features
`lag_1, lag_2, ..., lag_k` = xₜ₋₁, xₜ₋₂, ..., xₜ₋ₖ
→ Información del pasado inmediato ("¿qué pasó ayer?")

### Rolling Statistics
`roll_mean_7`, `roll_std_14`, `roll_max_30`...
→ Contexto histórico suavizado ("¿cómo estuvo la semana pasada en promedio?")

### Fourier Features
```
sin(2π·k·t/T),  cos(2π·k·t/T)  para k=1,...,K
```
→ Representación continua de la estacionalidad, mejor que dummies categóricas
→ K términos: K=1 (sinusoide simple), K=5+ (estacionalidad compleja)

---

## ETAPA 4 — Modelos

### Jerarquía de complejidad

```
Complejidad
     ↑
     │  🤖 Chronos-Bolt (Foundation Model, 2024)
     │  ⚡ LightGBM + lag features
     │  📐 SARIMA / SARIMAX
     │  📈 ETS (Holt-Winters)
     │  〰️ Seasonal Naïve
     └──────────────────────────────────→ Siempre más complejo ≠ mejor
```

### Naïve
`ŷ_{t+h} = y_t`
El baseline absoluto. Si no superas esto, algo está mal.

### Seasonal Naïve
`ŷ_{t+h} = y_{t+h-T}`  (repite el mismo período anterior)
Muy fuerte para datos con estacionalidad clara.

### ETS (Error-Trend-Seasonality)
Familia de modelos con suavizado exponencial.
Parámetros: α (nivel), β (tendencia), γ (estacionalidad).
Ganó el M3 Competition. Simple pero muy robusto.

### SARIMA(p,d,q)(P,D,Q,T)
`p,d,q`: componentes autoregresivo, integrado, media móvil
`P,D,Q`: sus equivalentes estacionales, `T`: período
Interpretable, muy usado en finanzas y economía.

### LightGBM + Lag Features
Gradient boosting con árboles de decisión.
Aprende relaciones no lineales entre lags.
**Ventajas**: rápido, no requiere estacionariedad, puede usar covariables.
**Desventaja**: forecast recursivo acumula error.

### Chronos-Bolt (Amazon, 2024)
Transformer preentrenado en 84,700+ series.
**Zero-shot**: sin reentrenamiento en tus datos.
Distribución completa de probabilidad, no solo punto.

### Ensemble
`ŷ_ensemble = (1/K) · Σ ŷ_modelo_k`
Combinar modelos suele mejorar la precisión (wisdom of crowds).

---

## ETAPA 5 — Evaluación Probabilística

### Por qué probabilístico es mejor
Un pronóstico puntual dice "venderás 100".
Un pronóstico probabilístico dice "hay 80% de probabilidad de que vendas entre 85 y 118".
→ La segunda información vale para gestión de inventario, riesgo, etc.

### CRPS (Continuous Ranked Probability Score)
```
CRPS(F,y) = E[|X-y|] - ½·E[|X-X'|]
```
- La métrica de referencia en GIFT-Eval, M4/M5 competitions
- Degeneración puntual: CRPS = MAE cuando F=Dirac(ŷ)
- Penaliza **tanto sesgo como mala calibración**

### Winkler Score (Interval Score)
```
IS_α = (hi-lo) + (2/α)(lo-y)·𝟙(y<lo) + (2/α)(y-hi)·𝟙(y>hi)
```
Para un IC del (1-α)%: suma ancho + penalización por misses.
Optimiza el trade-off sharpness vs. coverage.

### Calibración (Reliability Diagram)
Si prometes un IC del 80%, ¿se cumple el 80% de las veces?
- Curva = diagonal → calibración perfecta
- Curva por debajo de diagonal → intervalos demasiado estrechos (optimista)
- Curva por encima → intervalos demasiado anchos (conservador)

### Sharpness
Ancho del IC. **Más estrecho = más informativo**, pero debe estar calibrado.
Trade-off: ¿prefieres decir "entre -∞ y +∞" (100% cobertura, inútil) o "entre 95 y 105" (muy informativo pero puede fallar)?

### Ljung-Box Test (residuos)
H₀: no hay autocorrelación en residuos.
p < 0.05 → el modelo no ha capturado todo el patrón → prueba otro.

### Rolling Window Cross-Validation
```
Expanding window:
  [────────────────]·[horizon]  → CRPS₁
  [──────────────────]·[horizon] → CRPS₂
  [────────────────────]·[horizon] → CRPS₃
```
Promedio de CRPS sobre ventanas = estimación robusta del rendimiento real.

### Análisis de Residuos
- **Sesgo** (BIAS): ¿predice sistemáticamente alto o bajo?
- **Homocedasticidad**: ¿la varianza del error es constante en el tiempo?
- **Q-Q plot**: ¿los residuos son normales? (importante para ICs)
- **Residuos vs predicción**: ¿hay patrones no capturados?

---

## ETAPA 6 — Producción

### Pipeline recomendado
```
1. Datos → Preprocessing → Feature Engineering
2. Train (rolling CV para estimar rendimiento)
3. Selección de modelo (menor CRPS en CV)
4. Retrain con todos los datos
5. Forecast + IC
6. Monitoreo: detectar drift de datos
```

### Métricas de monitoreo en producción
- **MAPE Rolling**: ¿el error está aumentando con el tiempo?
- **Drift de distribución** (Kolmogorov-Smirnov): ¿los datos recientes difieren de los de entrenamiento?
- **Coverage drift**: ¿los ICs se cumplen con la frecuencia esperada?

---

## Referencias

| Recurso | Descripción |
|---|---|
| [FPP3 (Hyndman)](https://otexts.com/fpp3/) | El libro gratuito definitivo |
| [GIFT-Eval](https://arxiv.org/abs/2410.10393) | Benchmark de referencia (Salesforce 2024) |
| [Chronos](https://arxiv.org/abs/2403.07815) | Foundation model (Amazon 2024) |
| [Timer-S1](https://arxiv.org/abs/2603.04791) | SOTA actual 8.3B (THUML 2026) |
| [M4 Competition](https://www.sciencedirect.com/science/article/pii/S0169207019301128) | El benchmark clásico |
| [Nixtla](https://nixtla.io/) | Librería Python con todo |
| [GluonTS](https://gluonts.mxnet.io/) | AWS toolkit para TS |
"""

# ─────────────────────────────────────────────────────────────
#  UI
# ─────────────────────────────────────────────────────────────
CSS = """
#hdr{text-align:center;padding:18px 0 8px}
#hdr h1{background:linear-gradient(135deg,#60A5FA,#F87171,#A78BFA,#FBBF24);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;font-size:2.3em;font-weight:800}
.step-badge{display:inline-block;background:#1e293b;border:1px solid #334155;
  border-radius:8px;padding:4px 10px;font-size:0.85em;color:#94A3B8;margin-bottom:6px}
footer{visibility:hidden}
"""

with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="purple"),
    title="🎯 Temporal Arena Pro",
    css=CSS,
) as demo:

    gr.HTML("""
    <div id="hdr">
      <h1>🎯 Temporal Arena Pro</h1>
      <p style="color:#94A3B8;font-size:1.1em">
        Pipeline completo de Forecasting · Toda la cadena del dato<br>
        <b>Ingesta → EDA → Features → Modelos → Evaluación → Pronóstico</b>
      </p>
      <p style="color:#64748B;font-size:0.85em">
        por <a href="https://huggingface.co/JoseAndresLopez" target="_blank">JoseAndresLopez</a>
        &nbsp;|&nbsp; Chronos · LightGBM · ARIMA · ETS · CRPS · Calibración · Fan Charts
      </p>
    </div>
    """)

    with gr.Tabs():

        # ══ TAB 1: INGESTA ══════════════════════════════════════
        with gr.Tab("📥 1 · Ingesta de Datos"):
            gr.HTML('<div class="step-badge">PASO 1 — Elige tu fuente de datos</div>')
            with gr.Tabs():

                with gr.Tab("🌡️ Clima en vivo (OpenMeteo)"):
                    with gr.Row():
                        city_dd  = gr.Dropdown(list(CIUDADES.keys()), value="Madrid", label="Ciudad")
                        var_dd   = gr.Dropdown(list(METEO_VARS.keys()), value=list(METEO_VARS.keys())[0], label="Variable")
                        days_sl  = gr.Slider(90,1095,365,step=30,label="Días de histórico")
                    btn_weath = gr.Button("🌡️ Obtener datos reales (OpenMeteo)", variant="primary")

                with gr.Tab("📈 Finanzas (Yahoo Finance)"):
                    with gr.Row():
                        fin_dd   = gr.Dropdown(list(FINANCE_TICKERS.keys()), value="₿ Bitcoin (BTC-USD)", label="Activo")
                        fin_days = gr.Slider(90,1825,730,step=30,label="Días")
                    btn_fin  = gr.Button("📈 Obtener datos reales (Yahoo Finance)", variant="primary")

                with gr.Tab("📊 Datasets Benchmark"):
                    with gr.Row():
                        btn_etth1  = gr.Button("ETTh1 (GIFT-Eval · horario)", variant="secondary")
                        btn_ap     = gr.Button("AirPassengers (mensual clásico)", variant="secondary")
                        btn_en     = gr.Button("Consumo Energético (sintético)", variant="secondary")

                with gr.Tab("📁 Tu CSV"):
                    upload = gr.File(label="CSV con 2 columnas: fecha, valor", file_types=[".csv"])
                    btn_csv= gr.Button("Cargar CSV", variant="secondary")

            # Outputs de ingesta
            plot_preview = gr.Plot()
            with gr.Row():
                quality_md = gr.Markdown()
                stationary_md = gr.Markdown()

        # ══ TAB 2: EDA ══════════════════════════════════════════
        with gr.Tab("🔬 2 · Análisis Exploratorio"):
            gr.HTML('<div class="step-badge">PASO 2 — EDA: Entiende tu serie antes de modelar</div>')
            btn_eda = gr.Button("🔬 Ejecutar EDA completo", variant="primary", size="lg")
            with gr.Row():
                plot_decomp = gr.Plot()
            with gr.Row():
                plot_acf_p  = gr.Plot()
            with gr.Row():
                plot_dist   = gr.Plot()
                plot_period = gr.Plot()

        # ══ TAB 3: FEATURES ══════════════════════════════════════
        with gr.Tab("⚙️ 3 · Feature Engineering"):
            gr.HTML('<div class="step-badge">PASO 3 — Genera features para LightGBM (y visualiza su relevancia)</div>')
            with gr.Row():
                n_lags_sl   = gr.Slider(1,60,28,step=1,label="Número de lags")
                n_fourier_sl= gr.Slider(1,10,3,step=1,label="Términos Fourier")
            btn_feat = gr.Button("⚙️ Generar features", variant="primary")
            feat_plot= gr.Plot(label="Correlación features ↔ target")
            feat_md  = gr.Markdown()

        # ══ TAB 4: MODELOS ══════════════════════════════════════
        with gr.Tab("🤖 4 · Modelos & Pronóstico"):
            gr.HTML('<div class="step-badge">PASO 4 — Entrena todos los modelos y compara en backtesting</div>')
            with gr.Row():
                with gr.Column(scale=1,min_width=260):
                    h_sl     = gr.Slider(10,120,24,step=1,label="⏱️ Horizonte (pasos)")
                    model_dd = gr.Dropdown([
                        ("🚀 Chronos-Bolt Small (rápido)", "amazon/chronos-bolt-small"),
                        ("🎯 Chronos-Bolt Base (preciso)",  "amazon/chronos-bolt-base"),
                        ("🧪 Chronos T5-Small (original)",  "amazon/chronos-t5-small"),
                    ], value="amazon/chronos-bolt-small", label="🧠 Modelo Chronos")
                    nsamp_sl = gr.Slider(20,100,50,step=10,label="🎲 Muestras Monte Carlo")
                    gr.Markdown("**Selecciona modelos:**")
                    chk_chronos = gr.Checkbox(True,  label="🤖 Chronos-Bolt")
                    chk_ets     = gr.Checkbox(True,  label="📈 ETS (Holt-Winters)")
                    chk_arima   = gr.Checkbox(True,  label="📐 SARIMA")
                    chk_lgbm    = gr.Checkbox(True,  label="⚡ LightGBM (requiere features)")
                    chk_naive   = gr.Checkbox(True,  label="📏 Naïve")
                    chk_snaive  = gr.Checkbox(True,  label="🔄 Seasonal Naïve")
                    btn_models  = gr.Button("🚀 Entrenar y evaluar", variant="primary", size="lg")
                    gr.Markdown("> Primera vez: descarga Chronos (~200MB)")
                with gr.Column(scale=3):
                    fan_plot = gr.Plot()

            with gr.Row():
                board_md  = gr.Markdown()
            with gr.Row():
                calib_plot  = gr.Plot()
                sharp_plot  = gr.Plot()
            with gr.Row():
                resid_plot  = gr.Plot()
            with gr.Row():
                fi_plot     = gr.Plot()
            with gr.Row():
                lb_md       = gr.Markdown()

        # ══ TAB 5: ROLLING CV ════════════════════════════════════
        with gr.Tab("🔄 5 · Validación Cruzada"):
            gr.HTML('<div class="step-badge">PASO 5 — Rolling Window CV: la evaluación correcta para series temporales</div>')
            with gr.Row():
                h_roll  = gr.Slider(10,60,24,step=1,label="Horizonte")
                w_roll  = gr.Slider(3,10,5,step=1,label="Nº ventanas")
                model_roll = gr.Dropdown([
                    ("Chronos-Bolt Small","amazon/chronos-bolt-small"),
                    ("Chronos-Bolt Base", "amazon/chronos-bolt-base"),
                ], value="amazon/chronos-bolt-small",label="Modelo Chronos (opcional)")
            btn_roll = gr.Button("🔄 Ejecutar validación cruzada", variant="primary")
            roll_plot= gr.Plot()
            roll_md  = gr.Markdown()

        # ══ TAB 6: PRONÓSTICO FUTURO ════════════════════════════
        with gr.Tab("🔮 6 · Pronóstico Final"):
            gr.HTML('<div class="step-badge">PASO 6 — Pronóstico sobre el futuro real (todos los datos de entrenamiento)</div>')
            h_fut    = gr.Slider(5,120,24,step=1,label="Horizonte futuro")
            btn_fut  = gr.Button("🔮 Generar pronóstico final", variant="primary", size="lg")
            fut_plot = gr.Plot()
            fut_md   = gr.Markdown()
            csv_out  = gr.DownloadButton("⬇️ Descargar predicciones CSV",visible=False,variant="secondary")

        # ══ TAB 7: APRENDE ══════════════════════════════════════
        with gr.Tab("📚 7 · Aprende todo"):
            gr.Markdown(LEARN_MD)

    # ── Wiring ────────────────────────────────────────────────
    _eda_outs = [plot_preview, quality_md, stationary_md,
                 plot_decomp, plot_acf_p, plot_period]

    btn_weath.click(cb_load_weather, [city_dd,var_dd,days_sl], _eda_outs)
    btn_fin.click(  cb_load_finance, [fin_dd,fin_days],        _eda_outs)
    btn_etth1.click(cb_load_etth1,   [],                        _eda_outs)
    btn_ap.click(   cb_load_ap,      [],                        _eda_outs)
    btn_en.click(   cb_load_energia, [],                        _eda_outs)
    btn_csv.click(  cb_load_csv,     [upload],                  _eda_outs)

    btn_eda.click(cb_run_eda, [],
                  [plot_decomp, plot_acf_p, plot_dist, plot_period])

    btn_feat.click(cb_engineer, [n_lags_sl, n_fourier_sl],
                   [feat_plot, feat_md, gr.State(None)])

    btn_models.click(cb_run_models,
                     [h_sl,model_dd,nsamp_sl,
                      chk_chronos,chk_ets,chk_arima,chk_lgbm,chk_naive,chk_snaive],
                     [fan_plot,calib_plot,resid_plot,sharp_plot,
                      fi_plot,board_md,lb_md,gr.State(None),gr.State(None)])

    btn_roll.click(cb_rolling_cv, [h_roll,w_roll,model_roll], [roll_plot,roll_md])

    btn_fut.click(cb_future_forecast, [h_fut],
                  [fut_plot, fut_md, csv_out])

    # Carga inicial
    demo.load(fn=cb_load_ap, outputs=_eda_outs)


if __name__ == "__main__":
    demo.launch()
