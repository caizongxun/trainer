#!/usr/bin/env python3
"""colab_workflow_v24.py

V24 "The Reversal Specialist" (Specialized Reversal Features)

Goal: Break the 0.16 F1 score ceiling in Range Reversal tasks.
Strategy: Inject domain-specific "Reversal" features directly into GP.

New Features:
1. RSI Divergence (Slope_Price vs Slope_RSI)
2. Donchian Channel Position (Relative position in N-bar high-low range)
3. Stochastic KD (Fast %K, %D) - classic reversal indicator
4. Bollinger %B - relative position to bands

Architecture:
- Retains V23's Dual-Formula (Signal + Filter) structure.
- Retains Strict Bloat Control.

Run on Colab (GPU):
!pip install deap cupy-cuda12x && curl -s https://raw.githubusercontent.com/caizongxun/trainer/main/colab_workflow_v24.py | python3 - \
  --symbol BTCUSDT --interval 15m --pop_size 2000 \
  --max_gens 10000 --patience 100 --min_delta 1e-4 \
  --max_height 6 --max_nodes 80 \
  --use_gpu
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

XP = np

def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _as_float(x):
    if x is None: return None
    if isinstance(x, (tuple, list)): return float(x[0]) if len(x) else None
    if isinstance(x, np.ndarray): return float(x.flat[0]) if x.size else None
    return float(x)

def _to_py_scalar(x):
    try: return x.item()
    except: return float(x)

def _maybe_to_xp(arr, use_gpu: bool, dtype=None):
    if not use_gpu: return arr
    if dtype is None: dtype = XP.float32
    try: return XP.asarray(arr, dtype=dtype)
    except: return XP.asarray(arr)

def _xp_errstate(**kwargs):
    return XP.errstate(**kwargs)

# ------------------------------
# 1. Feature Engineering (Reversal Focus)
# ------------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    
    # --- Basic ---
    d["sma20"] = d["close"].rolling(20).mean()
    d["sma50"] = d["close"].rolling(50).mean()
    d["std20"] = d["close"].rolling(20).std()
    d["ema20"] = d["close"].ewm(span=20, adjust=False).mean()
    
    # RSI
    delta = d["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    rs = gain.ewm(alpha=1/14).mean() / (loss.ewm(alpha=1/14).mean() + 1e-12)
    d["rsi"] = 100.0 - (100.0 / (1.0 + rs))
    
    # ATR & ADX
    tr = pd.concat([d["high"]-d["low"], (d["high"]-d["close"].shift()).abs(), (d["low"]-d["close"].shift()).abs()], axis=1).max(axis=1)
    d["atr14"] = tr.rolling(14).mean()
    
    dmp = d["high"].diff().clip(lower=0)
    dmm = d["low"].diff().clip(upper=0).abs()
    plus_di = 100 * (dmp.ewm(alpha=1/14).mean() / (d["atr14"] + 1e-12))
    minus_di = 100 * (dmm.ewm(alpha=1/14).mean() / (d["atr14"] + 1e-12))
    dx = (abs(plus_di - minus_di) / (abs(plus_di + minus_di) + 1e-12)) * 100
    d["adx"] = dx.rolling(14).mean().fillna(0)
    
    d["bb_width"] = (4 * d["std20"]) / (d["sma20"].abs() + 1e-12)
    
    rng = (d["high"] - d["low"]) / (d["close"].abs() + 1e-12)
    d["range_z"] = (rng - rng.rolling(50).mean()) / (rng.rolling(50).std() + 1e-12)
    
    d["roc10"] = d["close"].pct_change(10)
    
    tp = (d["high"] + d["low"] + d["close"]) / 3.0
    mf = tp * d["volume"]
    pos_mf = mf.where(tp > tp.shift(1), 0.0)
    neg_mf = mf.where(tp < tp.shift(1), 0.0)
    mfr = pos_mf.rolling(14).sum() / (neg_mf.rolling(14).sum().abs() + 1e-12)
    d["mfi14"] = 100.0 - (100.0 / (1.0 + mfr))
    
    # --- NEW: Reversal Features ---
    
    # 1. Donchian Position (0~1)
    # 0 = at 20-day low, 1 = at 20-day high
    donchian_low = d["low"].rolling(20).min()
    donchian_high = d["high"].rolling(20).max()
    d["donchian_pos"] = (d["close"] - donchian_low) / (donchian_high - donchian_low + 1e-12)
    
    # 2. Stochastic KD (Fast)
    lowest_14 = d["low"].rolling(14).min()
    highest_14 = d["high"].rolling(14).max()
    d["stoch_k"] = 100 * ((d["close"] - lowest_14) / (highest_14 - lowest_14 + 1e-12))
    d["stoch_d"] = d["stoch_k"].rolling(3).mean()
    
    # 3. RSI Divergence Approximation
    # Slope of Price (5 bars) vs Slope of RSI (5 bars)
    # If Price Slope < 0 and RSI Slope > 0 => Bullish Divergence
    # We provide the raw slopes, let GP find the divergence condition
    # Linear regression slope approximation: (y - y_mean) * (x - x_mean)
    # Simplified: just pct_change(5) or diff(5)
    d["slope_price_5"] = d["close"].diff(5)
    d["slope_rsi_5"] = d["rsi"].diff(5)
    
    # 4. Bollinger %B
    # Position relative to bands. >1 = above upper, <0 = below lower
    bb_upper = d["sma20"] + 2*d["std20"]
    bb_lower = d["sma20"] - 2*d["std20"]
    d["bb_pct_b"] = (d["close"] - bb_lower) / (bb_upper - bb_lower + 1e-12)

    d = d.dropna().reset_index(drop=True)
    return d

# ------------------------------
# 2. Regime Classification
# ------------------------------
def classify_regimes(df: pd.DataFrame) -> np.ndarray:
    X = df[["adx", "range_z"]].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    gmm = GaussianMixture(n_components=2, random_state=42)
    labels = gmm.fit_predict(X_scaled)
    if df.loc[labels==1, "adx"].mean() > df.loc[labels==0, "adx"].mean():
        return labels 
    return 1 - labels

# ------------------------------
# 3. Labeling Targets
# ------------------------------
def label_targets(df: pd.DataFrame) -> pd.DataFrame:
    lookback = 12
    # Task A: Range Reversal
    df["pivot_low"] = df["low"].rolling(window=lookback*2+1, center=True).min() == df["low"]
    df["target_range_buy"] = (df["pivot_low"] & (df["regime"] == 0)).astype(int)
    
    # Task B: Trend Start
    df["regime_switch"] = (df["regime"] == 1) & (df["regime"].shift(1) == 0)
    df["future_ret"] = df["close"].shift(-10) / df["close"] - 1
    df["target_trend_start"] = (df["regime_switch"] & (df["future_ret"] > 0.02)).astype(int)
    
    return df

# ------------------------------
# 4. GP Primitives
# ------------------------------
def add(a, b): return XP.add(a, b)
def sub(a, b): return XP.subtract(a, b)
def mul(a, b): return XP.multiply(a, b)
def neg(a): return XP.negative(a)
def absv(a): return XP.abs(a)
def protectedDiv(left, right):
    with _xp_errstate(divide="ignore", invalid="ignore"):
        x = XP.divide(left, right)
        x = XP.where(XP.isinf(x), 1, x)
        x = XP.where(XP.isnan(x), 1, x)
    return x
def if_then(condition, out1, out2):
    return XP.where(condition > 0, out1, out2)

pset = gp.PrimitiveSet("MAIN", 20)
pset.addPrimitive(add, 2, name="add")
pset.addPrimitive(sub, 2, name="sub")
pset.addPrimitive(mul, 2, name="mul")
pset.addPrimitive(protectedDiv, 2, name="div")
pset.addPrimitive(neg, 1, name="neg")
pset.addPrimitive(absv, 1, name="abs")
pset.addPrimitive(if_then, 3, name="if_gt_0")

arg_names = [
    "Close", "Open", "High", "Low", "Volume", "RSI", "SMA50", "EMA20", "ATR14", 
    "BBWidth", "ROC10", "MFI14", "ADX", "RangeZ",
    "DonchianPos", "StochK", "StochD", "SlopePrice", "SlopeRSI", "BBPctB"
]
for i, n in enumerate(arg_names): pset.renameArguments(**{f"ARG{i}": n})

# Multi-Tree Individual
if not hasattr(creator, "FitnessMax"): creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"): creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr_signal", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("expr_filter", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)

def init_individual(icls, content): return icls(content)

toolbox.register("individual", tools.initCycle, creator.Individual, (lambda: gp.PrimitiveTree(toolbox.expr_signal()), lambda: gp.PrimitiveTree(toolbox.expr_filter())), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def cxDualTree(ind1, ind2):
    if random.random() < 0.5: gp.cxOnePoint(ind1[0], ind2[0])
    else: gp.cxOnePoint(ind1[1], ind2[1])
    return ind1, ind2

def mutDualTree(ind, expr):
    if random.random() < 0.5: gp.mutUniform(ind[0], expr, pset)
    else: gp.mutUniform(ind[1], expr, pset)
    return ind,

toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", cxDualTree)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", mutDualTree, expr=toolbox.expr_mut)

GLOBAL_INPUTS = {}
GLOBAL_TARGET = None

def eval_dual_formula(individual):
    if len(individual[0]) > 100 or len(individual[1]) > 100: return (0.0,)
    func_sig = toolbox.compile(expr=individual[0])
    func_fil = toolbox.compile(expr=individual[1])
    try:
        args = [GLOBAL_INPUTS[k] for k in arg_names]
        out_sig = func_sig(*args)
        out_fil = func_fil(*args)
        final_signal = ((out_sig > 0) & (out_fil > 0)).astype(XP.int32)
        
        hits = _to_py_scalar(XP.sum((final_signal == 1) & (GLOBAL_TARGET == 1)))
        total_signals = _to_py_scalar(XP.sum(final_signal == 1))
        
        if total_signals == 0: return (0.0,)
        
        precision = float(hits) / float(total_signals)
        if total_signals < 10: precision = 0.0
        
        target_ones = _to_py_scalar(XP.sum(GLOBAL_TARGET == 1))
        recall = float(hits) / float(target_ones + 1e-9)
        
        beta = 0.5 # Precision weight
        f_score = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall + 1e-9)
        return (float(f_score),)
    except: return (0.0,)

toolbox.register("evaluate", eval_dual_formula)

def algorithms_eaSimple(pop, toolbox, cxpb, mutpb, ngen, stats, halloffame, verbose, patience, min_delta):
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])
    
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    fits = map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fits): ind.fitness.values = fit
    if halloffame: halloffame.update(pop)
    
    record = stats.compile(pop)
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose: print(logbook.stream)
    
    best_so_far = _as_float(record.get("max", 0.0))
    stall = 0
    best_gen = 0
    
    for gen in range(1, ngen+1):
        offspring = toolbox.select(pop, len(pop))
        offspring = [toolbox.clone(ind) for ind in offspring]
        
        for i in range(1, len(offspring), 2):
            if random.random() < cxpb:
                toolbox.mate(offspring[i-1], offspring[i])
                del offspring[i-1].fitness.values, offspring[i].fitness.values
        for i in range(len(offspring)):
            if random.random() < mutpb:
                toolbox.mutate(offspring[i])
                del offspring[i].fitness.values
                
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fits = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fits): ind.fitness.values = fit
        
        if halloffame: halloffame.update(offspring)
        pop[:] = offspring
        
        record = stats.compile(pop)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose: print(logbook.stream)
        
        current_best = _as_float(record.get("max", 0.0))
        if current_best > best_so_far + min_delta:
            best_so_far = current_best
            best_gen = gen
            stall = 0
        else:
            stall += 1
            
        if patience and stall >= patience:
            print(f"EARLY STOP: gen={gen}, best={best_so_far:.6f}")
            logbook.stopped_early = True
            logbook.best_gen = best_gen
            break
            
    logbook.best_score = best_so_far
    if not hasattr(logbook, 'best_gen'): logbook.best_gen = best_gen
    return pop, logbook

def evolve_for_task(task_name, target_array, inputs, pop_size=50, gens=5, patience=0, min_delta=0.0, use_gpu=False):
    print(f"\n--- Evolving for Task: {task_name} (Dual-Formula + Reversal Features) ---")
    global GLOBAL_INPUTS, GLOBAL_TARGET
    GLOBAL_INPUTS = inputs
    GLOBAL_TARGET = target_array
    
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    
    pop, log = algorithms_eaSimple(pop, toolbox, 0.5, 0.2, gens, stats, hof, True, patience, min_delta)
    
    best = hof[0]
    best_score = float(best.fitness.values[0])
    
    print(f"Best Signal Formula: {best[0]}")
    print(f"Best Filter Formula: {best[1]}")
    print(f"Best Weighted F-Score: {best_score:.6f}")
    
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
    return f"SIGNAL: {best[0]} | FILTER: {best[1]}", best_score, meta

def main():
    global XP, GLOBAL_INPUTS, GLOBAL_TARGET
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", type=str, default="BTCUSDT")
    p.add_argument("--interval", type=str, default="15m")
    p.add_argument("--max_gens", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--min_delta", type=float, default=1e-4)
    p.add_argument("--pop_size", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_height", type=int, default=6)
    p.add_argument("--max_nodes", type=int, default=80)
    p.add_argument("--use_gpu", action="store_true")
    args = p.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    if args.use_gpu:
        try:
            import cupy as cp
            XP = cp
            print("[GPU] CuPy Enabled")
        except:
            print("[GPU] Failed, using NumPy")
            XP = np
    else: XP = np
    
    from huggingface_hub import snapshot_download
    path = snapshot_download(repo_id="zongowo111/cpb-models", repo_type="dataset")
    csv_file = None
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith(".csv") and args.symbol in f and args.interval in f:
                csv_file = os.path.join(root, f)
                break
    df = pd.read_csv(csv_file)
    time_col = next(c for c in df.columns if "time" in c)
    if pd.api.types.is_numeric_dtype(df[time_col]):
        df["open_time"] = pd.to_datetime(df[time_col], unit="ms", utc=True)
    else:
        df["open_time"] = pd.to_datetime(df[time_col], utc=True)
    df = df.sort_values("open_time").reset_index(drop=True)
    
    print("[2/5] Feature Engineering (Reversal Focus)...")
    df = add_features(df)
    
    print("[3/5] Classifying Regimes (GMM)...")
    df["regime"] = classify_regimes(df)
    
    print("[4/5] Labeling Targets...")
    df = label_targets(df)
    
    inputs = {k: _maybe_to_xp(df[k.lower() if k not in ["BBWidth", "RangeZ", "DonchianPos", "StochK", "StochD", "SlopePrice", "SlopeRSI", "BBPctB"] else 
                              "bb_width" if k=="BBWidth" else 
                              "range_z" if k=="RangeZ" else
                              "donchian_pos" if k=="DonchianPos" else
                              "stoch_k" if k=="StochK" else
                              "stoch_d" if k=="StochD" else
                              "slope_price_5" if k=="SlopePrice" else
                              "slope_rsi_5" if k=="SlopeRSI" else
                              "bb_pct_b" if k=="BBPctB"].values, args.use_gpu) for k in arg_names}
    
    out_dir = f"./all_models/models_v24/{args.symbol}"
    _safe_mkdir(out_dir)
    report_file = os.path.join(out_dir, "reversal_report.txt")
    
    with open(report_file, "w") as f:
        f.write(f"=== THE REVERSAL SPECIALIST REPORT (V24) ({args.symbol}) ===\n")
        
        # Task A
        target = _maybe_to_xp(df["target_range_buy"].values.astype(np.int32), args.use_gpu, XP.int32)
        best, score, meta = evolve_for_task("Range Reversal", target, inputs, args.pop_size, args.max_gens, args.patience, args.min_delta, args.use_gpu)
        f.write(f"Task: Range Reversal\n{best}\nWeighted Score: {score:.6f}\n\n")
        
        # Task B
        target = _maybe_to_xp(df["target_trend_start"].values.astype(np.int32), args.use_gpu, XP.int32)
        best, score, meta = evolve_for_task("Trend Start", target, inputs, args.pop_size, args.max_gens, args.patience, args.min_delta, args.use_gpu)
        f.write(f"Task: Trend Start\n{best}\nWeighted Score: {score:.6f}\n\n")
        
    print(f"\n[5/5] Done! Report saved to {report_file}")

if __name__ == "__main__":
    main()
