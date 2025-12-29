#!/usr/bin/env python3
"""colab_workflow_v20.py

V20 "The Alpha Factory" (Evolutionary Regime-Specific Formula Discovery)

New in this update (still V20)
- Adds optional GPU acceleration for GP evaluation using CuPy:
  - --use_gpu
  - When enabled, GP primitives run on CuPy arrays (GPU), while data loading/feature engineering and GMM remain on CPU.

V20 features
- Adds more "building blocks" (domain features) as GP terminals:
  - EMA20, ATR14, BBWidth, ROC10, MFI14, ADX, RangeZ
- Keeps V19 early stopping ("train until no longer improves"):
  - --max_gens, --patience, --min_delta
- Optional bloat control:
  - --max_height, --max_nodes

Run on Colab (GPU)
!pip install deap cupy-cuda12x && curl -s https://raw.githubusercontent.com/caizongxun/trainer/main/colab_workflow_v20.py | python3 - \
  --symbol BTCUSDT --interval 15m --pop_size 2000 \
  --max_gens 10000 --patience 100 --min_delta 1e-4 \
  --use_gpu

Artifacts
- text : ./all_models/models_v20/{symbol}/alpha_factory_report.txt
"""

import os
import argparse
import random
import operator
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from deap import base, creator, tools, gp

# Backend (NumPy by default; optionally switched to CuPy in main via --use_gpu)
XP = np


def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _as_float(x):
    """DEAP stats sometimes returns tuples like (0.123,)"""
    if x is None:
        return None
    if isinstance(x, (tuple, list)):
        return float(x[0]) if len(x) else None
    if isinstance(x, np.ndarray):
        return float(x.flat[0]) if x.size else None
    return float(x)


def _to_py_scalar(x):
    """Convert numpy/cupy scalar to python float/int."""
    try:
        return x.item()
    except Exception:
        return float(x)


def _maybe_to_xp(arr, use_gpu: bool, dtype=None):
    if not use_gpu:
        return arr
    # Cast to float32 by default for VRAM efficiency unless caller requests otherwise.
    if dtype is None:
        dtype = XP.float32
    try:
        return XP.asarray(arr, dtype=dtype)
    except Exception:
        return XP.asarray(arr)


def _xp_errstate(**kwargs):
    # Both numpy and cupy provide errstate.
    return XP.errstate(**kwargs)


# ------------------------------
# 1. Feature Engineering (CPU)
# ------------------------------

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # Basic moving stats
    d["sma20"] = d["close"].rolling(20).mean()
    d["sma50"] = d["close"].rolling(50).mean()
    d["std20"] = d["close"].rolling(20).std()

    # EMA
    d["ema20"] = d["close"].ewm(span=20, adjust=False).mean()

    # RSI(14)
    delta = d["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    d["rsi"] = 100.0 - (100.0 / (1.0 + rs))

    # True Range / ATR(14)
    tr1 = d["high"] - d["low"]
    tr2 = (d["high"] - d["close"].shift()).abs()
    tr3 = (d["low"] - d["close"].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    d["atr14"] = tr.rolling(14).mean()

    # ADX (for regime + as a GP terminal)
    dmp = d["high"].diff()
    dmm = d["low"].diff()
    dmp[dmp < 0] = 0
    dmm[dmm > 0] = 0
    atr_for_adx = d["atr14"]
    plus_di = 100 * (dmp.ewm(alpha=1 / 14).mean() / (atr_for_adx + 1e-12))
    minus_di = 100 * (dmm.abs().ewm(alpha=1 / 14).mean() / (atr_for_adx + 1e-12))
    dx = (abs(plus_di - minus_di) / (abs(plus_di + minus_di) + 1e-12)) * 100
    d["adx"] = dx.rolling(14).mean().fillna(0)

    # Bollinger Bands width (20, 2)
    bb_mid = d["sma20"]
    bb_upper = bb_mid + 2.0 * d["std20"]
    bb_lower = bb_mid - 2.0 * d["std20"]
    d["bb_width"] = (bb_upper - bb_lower) / (bb_mid.abs() + 1e-12)

    # Volatility Z (range and its z-score)
    d["range"] = (d["high"] - d["low"]) / (d["close"].abs() + 1e-12)
    d["range_z"] = (d["range"] - d["range"].rolling(50).mean()) / (d["range"].rolling(50).std() + 1e-12)

    # ROC(10)
    d["roc10"] = d["close"].pct_change(10)

    # MFI(14)
    tp = (d["high"] + d["low"] + d["close"]) / 3.0
    mf = tp * d["volume"]
    pos_mf = mf.where(tp > tp.shift(1), 0.0)
    neg_mf = mf.where(tp < tp.shift(1), 0.0)
    pos_sum = pos_mf.rolling(14).sum()
    neg_sum = neg_mf.rolling(14).sum().abs()
    mfr = pos_sum / (neg_sum + 1e-12)
    d["mfi14"] = 100.0 - (100.0 / (1.0 + mfr))

    d = d.dropna().reset_index(drop=True)
    return d


# ------------------------------
# 2. Regime Classification (CPU)
# ------------------------------

def classify_regimes(df: pd.DataFrame) -> np.ndarray:
    X = df[["adx", "range_z"]].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    gmm = GaussianMixture(n_components=2, random_state=42)
    labels = gmm.fit_predict(X_scaled)

    mean_adx_0 = df.loc[labels == 0, "adx"].mean()
    mean_adx_1 = df.loc[labels == 1, "adx"].mean()

    if mean_adx_1 > mean_adx_0:
        return labels  # 1=Trend, 0=Range
    return 1 - labels  # Flip so 1=Trend


# ------------------------------
# 3. Labeling Targets (CPU)
# ------------------------------

def label_targets(df: pd.DataFrame) -> pd.DataFrame:
    lookback = 12
    df["pivot_low"] = df["low"].rolling(window=lookback * 2 + 1, center=True).min() == df["low"]
    df["pivot_high"] = df["high"].rolling(window=lookback * 2 + 1, center=True).max() == df["high"]

    # Task A: Range Reversal (Buy at Pivot Low in Range)
    df["target_range_buy"] = (df["pivot_low"] & (df["regime"] == 0)).astype(int)

    # Task B: Trend Start (Buy when Trend starts)
    df["regime_switch"] = (df["regime"] == 1) & (df["regime"].shift(1) == 0)
    df["future_ret"] = df["close"].shift(-10) / df["close"] - 1
    df["target_trend_start"] = (df["regime_switch"] & (df["future_ret"] > 0.02)).astype(int)  # >2% pump

    # Task C: Trend End (Sell at Pivot High in Trend)
    df["target_trend_end"] = (df["pivot_high"] & (df["regime"] == 1)).astype(int)

    return df


# ------------------------------
# 4. GP primitives (CPU/GPU depending on XP)
# ------------------------------

def add(a, b):
    return XP.add(a, b)


def sub(a, b):
    return XP.subtract(a, b)


def mul(a, b):
    return XP.multiply(a, b)


def neg(a):
    return XP.negative(a)


def absv(a):
    return XP.abs(a)


def protectedDiv(left, right):
    with _xp_errstate(divide="ignore", invalid="ignore"):
        x = XP.divide(left, right)
        # Replace inf/nan with 1 (vectorized)
        x = XP.where(XP.isinf(x), 1, x)
        x = XP.where(XP.isnan(x), 1, x)
    return x


def if_then(condition, out1, out2):
    return XP.where(condition > 0, out1, out2)


# ------------------------------
# 5. GP setup
# ------------------------------

# Terminals ("building blocks")
# 14 inputs
# 0: Close
# 1: Open
# 2: High
# 3: Low
# 4: Volume
# 5: RSI
# 6: SMA50
# 7: EMA20
# 8: ATR14
# 9: BBWidth
# 10: ROC10
# 11: MFI14
# 12: ADX
# 13: RangeZ
pset = gp.PrimitiveSet("MAIN", 14)
pset.addPrimitive(add, 2, name="add")
pset.addPrimitive(sub, 2, name="sub")
pset.addPrimitive(mul, 2, name="mul")
pset.addPrimitive(protectedDiv, 2, name="div")
pset.addPrimitive(neg, 1, name="neg")
pset.addPrimitive(absv, 1, name="abs")
pset.addPrimitive(if_then, 3, name="if_gt_0")

pset.renameArguments(ARG0="Close")
pset.renameArguments(ARG1="Open")
pset.renameArguments(ARG2="High")
pset.renameArguments(ARG3="Low")
pset.renameArguments(ARG4="Volume")
pset.renameArguments(ARG5="RSI")
pset.renameArguments(ARG6="SMA50")
pset.renameArguments(ARG7="EMA20")
pset.renameArguments(ARG8="ATR14")
pset.renameArguments(ARG9="BBWidth")
pset.renameArguments(ARG10="ROC10")
pset.renameArguments(ARG11="MFI14")
pset.renameArguments(ARG12="ADX")
pset.renameArguments(ARG13="RangeZ")

# Creator is global and cannot be re-created in same process.
if not hasattr(creator, "FitnessMax"):
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)


toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

# Global Data for Eval (NumPy or CuPy arrays)
GLOBAL_INPUTS = {}
GLOBAL_TARGET = None


def eval_formula(individual):
    func = toolbox.compile(expr=individual)
    try:
        keys = [
            "Close",
            "Open",
            "High",
            "Low",
            "Volume",
            "RSI",
            "SMA50",
            "EMA20",
            "ATR14",
            "BBWidth",
            "ROC10",
            "MFI14",
            "ADX",
            "RangeZ",
        ]
        args = [GLOBAL_INPUTS[k] for k in keys]
        output = func(*args)

        # Convert output to signal (threshold > 0)
        signal = (output > 0).astype(XP.int32)

        hits = _to_py_scalar(XP.sum((signal == 1) & (GLOBAL_TARGET == 1)))
        total_signals = _to_py_scalar(XP.sum(signal == 1))
        if total_signals == 0:
            return (0.0,)

        precision = float(hits) / float(total_signals)

        # Penalize if too few signals
        if total_signals < 10:
            precision *= 0.1

        target_ones = _to_py_scalar(XP.sum(GLOBAL_TARGET == 1))
        recall = float(hits) / float(target_ones + 1e-9)
        f1 = 2.0 * (precision * recall) / (precision + recall + 1e-9)

        return (float(f1),)
    except Exception:
        return (0.0,)


toolbox.register("evaluate", eval_formula)


def algorithms_eaSimple(
    population,
    toolbox,
    cxpb,
    mutpb,
    ngen,
    stats=None,
    halloffame=None,
    verbose=__debug__,
    patience=0,
    min_delta=0.0,
):
    """eaSimple with optional early stopping.

    Early stopping:
    - Track stats['max'].
    - Stop when no improvement >= min_delta for `patience` consecutive generations.
    """
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    best_so_far = _as_float(record.get("max", 0.0)) if stats else 0.0
    best_gen = 0
    stall = 0
    stopped_early = False
    stop_reason = ""

    for gen in range(1, ngen + 1):
        offspring = toolbox.select(population, len(population))
        offspring = [toolbox.clone(ind) for ind in offspring]

        for i in range(1, len(offspring), 2):
            if random.random() < cxpb:
                toolbox.mate(offspring[i - 1], offspring[i])
                del offspring[i - 1].fitness.values
                del offspring[i].fitness.values

        for i in range(len(offspring)):
            if random.random() < mutpb:
                toolbox.mutate(offspring[i])
                del offspring[i].fitness.values

        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(offspring)

        population[:] = offspring
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        if stats and patience and patience > 0:
            current_best = _as_float(record.get("max", None))
            if current_best is not None and current_best > best_so_far + float(min_delta):
                best_so_far = current_best
                best_gen = gen
                stall = 0
            else:
                stall += 1

            if stall >= patience:
                stopped_early = True
                stop_reason = f"No improvement for {patience} generations (min_delta={min_delta})."
                if verbose:
                    print(
                        f"EARLY STOP at gen={gen} best={best_so_far:.6f} best_gen={best_gen} | {stop_reason}"
                    )
                break

    logbook.best_score = best_so_far
    logbook.best_gen = best_gen
    logbook.stopped_early = stopped_early
    logbook.stop_reason = stop_reason

    return population, logbook


def evolve_for_task(
    task_name,
    target_array,
    inputs,
    pop_size=50,
    gens=5,
    patience=0,
    min_delta=0.0,
    use_gpu=False,
):
    print(f"--- Evolving for Task: {task_name} ---")
    global GLOBAL_INPUTS, GLOBAL_TARGET

    # Move arrays to XP once per task (avoid host<->device copy every eval)
    GLOBAL_INPUTS = {}
    for k, v in inputs.items():
        GLOBAL_INPUTS[k] = _maybe_to_xp(v, use_gpu=use_gpu, dtype=None)

    # Targets as int32 on the same backend
    GLOBAL_TARGET = _maybe_to_xp(target_array.astype(np.int32), use_gpu=use_gpu, dtype=XP.int32)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)

    pop, log = algorithms_eaSimple(
        pop,
        toolbox,
        cxpb=0.5,
        mutpb=0.2,
        ngen=gens,
        stats=stats,
        halloffame=hof,
        verbose=True,
        patience=patience,
        min_delta=min_delta,
    )

    best = hof[0]
    best_score = float(best.fitness.values[0])

    print(f"Best Formula: {best}")
    print(f"Best Score: {best_score:.6f}")

    meta = {
        "best_gen": int(getattr(log, "best_gen", 0)),
        "best_score": float(getattr(log, "best_score", best_score)),
        "stopped_early": bool(getattr(log, "stopped_early", False)),
        "stop_reason": str(getattr(log, "stop_reason", "")),
        "gens_limit": int(gens),
        "patience": int(patience),
        "min_delta": float(min_delta),
        "use_gpu": bool(use_gpu),
        "backend": "cupy" if use_gpu else "numpy",
    }

    return str(best), best_score, meta


# ------------------------------
# Main
# ------------------------------

def main():
    global XP

    p = argparse.ArgumentParser()
    p.add_argument("--symbol", type=str, default="BTCUSDT")
    p.add_argument("--interval", type=str, default="15m")

    # Backward compatible param (V18/V19 style). Used when --max_gens is not provided.
    p.add_argument("--generations", type=int, default=10)

    # "Run until no longer improves" with a safe upper bound + early stopping.
    p.add_argument("--max_gens", type=int, default=None)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--min_delta", type=float, default=1e-6)

    p.add_argument("--pop_size", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)

    # Optional bloat control
    p.add_argument("--max_height", type=int, default=17)
    p.add_argument("--max_nodes", type=int, default=200)

    # GPU
    p.add_argument("--use_gpu", action="store_true")

    args = p.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Switch backend if requested
    if args.use_gpu:
        try:
            import cupy as cp

            XP = cp
            try:
                dev = cp.cuda.runtime.getDevice()
                props = cp.cuda.runtime.getDeviceProperties(dev)
                name = props.get("name", b"?")
                name = name.decode("utf-8") if isinstance(name, (bytes, bytearray)) else str(name)
                print(f"[GPU] Using CuPy on device {dev}: {name}")
            except Exception:
                print("[GPU] Using CuPy (device info unavailable)")
        except Exception as e:
            print(f"[GPU] Failed to enable CuPy, fallback to NumPy. Error: {e}")
            XP = np
            args.use_gpu = False
    else:
        XP = np

    max_gens = args.max_gens if args.max_gens is not None else args.generations

    # Apply bloat control limits
    if args.max_height and args.max_height > 0:
        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=args.max_height))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=args.max_height))
    if args.max_nodes and args.max_nodes > 0:
        toolbox.decorate("mate", gp.staticLimit(key=len, max_value=args.max_nodes))
        toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=args.max_nodes))

    print("[1/5] Loading Data...")
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id="zongowo111/cpb-models", repo_type="dataset")

    csv_file = None
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith(".csv") and args.symbol in f and args.interval in f:
                csv_file = os.path.join(root, f)
                break
        if csv_file:
            break

    if not csv_file:
        raise FileNotFoundError(f"No CSV found for symbol={args.symbol}, interval={args.interval}")

    df = pd.read_csv(csv_file)

    # Parse time
    time_col = next((c for c in ["open_time", "opentime", "timestamp", "time", "date"] if c in df.columns), None)
    if time_col is None:
        raise ValueError(f"No time column found in CSV columns: {list(df.columns)}")

    if pd.api.types.is_numeric_dtype(df[time_col]):
        ts = df[time_col].astype("int64")
        unit = "ms" if ts.median() > 10**12 else "s"
        df["open_time"] = pd.to_datetime(ts, unit=unit, utc=True)
    else:
        df["open_time"] = pd.to_datetime(df[time_col], utc=True, errors="coerce")

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().sort_values("open_time").reset_index(drop=True)

    print("[2/5] Feature Engineering...")
    df = add_features(df)

    print("[3/5] Classifying Regimes (GMM)...")
    df["regime"] = classify_regimes(df)
    print(f"Range Candles: {sum(df['regime'] == 0)}, Trend Candles: {sum(df['regime'] == 1)}")

    print("[4/5] Labeling Targets...")
    df = label_targets(df)

    inputs = {
        "Close": df["close"].values,
        "Open": df["open"].values,
        "High": df["high"].values,
        "Low": df["low"].values,
        "Volume": df["volume"].values,
        "RSI": df["rsi"].values,
        "SMA50": df["sma50"].values,
        "EMA20": df["ema20"].values,
        "ATR14": df["atr14"].values,
        "BBWidth": df["bb_width"].values,
        "ROC10": df["roc10"].values,
        "MFI14": df["mfi14"].values,
        "ADX": df["adx"].values,
        "RangeZ": df["range_z"].values,
    }

    out_dir = f"./all_models/models_v20/{args.symbol}"
    _safe_mkdir(out_dir)
    report_file = os.path.join(out_dir, "alpha_factory_report.txt")

    with open(report_file, "w") as f:
        f.write(f"=== ALPHA FACTORY REPORT (V20) ({args.symbol}) ===\n\n")
        f.write(f"interval: {args.interval}\n")
        f.write(f"pop_size: {args.pop_size}\n")
        f.write(f"max_gens: {max_gens}\n")
        f.write(f"patience: {args.patience}\n")
        f.write(f"min_delta: {args.min_delta}\n")
        f.write(f"seed: {args.seed}\n")
        f.write(f"max_height: {args.max_height}\n")
        f.write(f"max_nodes: {args.max_nodes}\n")
        f.write(f"use_gpu: {args.use_gpu}\n\n")

        # Task A
        best_formula, score, meta = evolve_for_task(
            "Range Reversal (Buy Low)",
            df["target_range_buy"].values,
            inputs,
            pop_size=args.pop_size,
            gens=max_gens,
            patience=args.patience,
            min_delta=args.min_delta,
            use_gpu=args.use_gpu,
        )
        f.write(
            "Task: Range Reversal\n"
            f"Formula: {best_formula}\n"
            f"Score (F1): {score:.6f}\n"
            f"best_gen: {meta['best_gen']}\n"
            f"stopped_early: {meta['stopped_early']}\n"
            f"stop_reason: {meta['stop_reason']}\n\n"
        )

        # Task B
        best_formula, score, meta = evolve_for_task(
            "Trend Start (Breakout)",
            df["target_trend_start"].values,
            inputs,
            pop_size=args.pop_size,
            gens=max_gens,
            patience=args.patience,
            min_delta=args.min_delta,
            use_gpu=args.use_gpu,
        )
        f.write(
            "Task: Trend Start\n"
            f"Formula: {best_formula}\n"
            f"Score (F1): {score:.6f}\n"
            f"best_gen: {meta['best_gen']}\n"
            f"stopped_early: {meta['stopped_early']}\n"
            f"stop_reason: {meta['stop_reason']}\n\n"
        )

        # Task C
        best_formula, score, meta = evolve_for_task(
            "Trend End (Climax)",
            df["target_trend_end"].values,
            inputs,
            pop_size=args.pop_size,
            gens=max_gens,
            patience=args.patience,
            min_delta=args.min_delta,
            use_gpu=args.use_gpu,
        )
        f.write(
            "Task: Trend End\n"
            f"Formula: {best_formula}\n"
            f"Score (F1): {score:.6f}\n"
            f"best_gen: {meta['best_gen']}\n"
            f"stopped_early: {meta['stopped_early']}\n"
            f"stop_reason: {meta['stop_reason']}\n\n"
        )

    print(f"\n[5/5] Done! Report saved to {report_file}")


if __name__ == "__main__":
    main()
