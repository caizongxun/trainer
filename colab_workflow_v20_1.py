#!/usr/bin/env python3
"""colab_workflow_v20_1.py

V20.1 "The Range Hunter" (Range-Specific Evolution)

This script focuses EXCLUSIVELY on evolving formulas for:
- Task A: Range Reversal (Buy Low in Range)

Features:
- V20 Core (Rich Indicators, Bloat Control, GPU Support)
- Specialized reporting for Range strategies.

Run on Colab (GPU):
!pip install deap cupy-cuda12x && curl -s https://raw.githubusercontent.com/caizongxun/trainer/main/colab_workflow_v20_1.py | python3 - \
  --symbol BTCUSDT --interval 15m --pop_size 2000 \
  --max_gens 10000 --patience 100 --min_delta 1e-4 \
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

# Backend (NumPy by default; optionally switched to CuPy in main via --use_gpu)
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
# 1. Feature Engineering
# ------------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["sma20"] = d["close"].rolling(20).mean()
    d["sma50"] = d["close"].rolling(50).mean()
    d["std20"] = d["close"].rolling(20).std()
    d["ema20"] = d["close"].ewm(span=20, adjust=False).mean()
    
    delta = d["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    rs = gain.ewm(alpha=1/14).mean() / (loss.ewm(alpha=1/14).mean() + 1e-12)
    d["rsi"] = 100.0 - (100.0 / (1.0 + rs))
    
    tr = pd.concat([d["high"]-d["low"], (d["high"]-d["close"].shift()).abs(), (d["low"]-d["close"].shift()).abs()], axis=1).max(axis=1)
    d["atr14"] = tr.rolling(14).mean()
    
    dmp = d["high"].diff().clip(lower=0)
    dmm = d["low"].diff().clip(upper=0).abs()
    plus_di = 100 * (dmp.ewm(alpha=1/14).mean() / (d["atr14"] + 1e-12))
    minus_di = 100 * (dmm.ewm(alpha=1/14).mean() / (d["atr14"] + 1e-12))
    dx = (abs(plus_di - minus_di) / (abs(plus_di + minus_di) + 1e-12)) * 100
    d["adx"] = dx.rolling(14).mean().fillna(0)
    
    bb_mid = d["sma20"]
    d["bb_width"] = (4 * d["std20"]) / (bb_mid.abs() + 1e-12)
    
    rng = (d["high"] - d["low"]) / (d["close"].abs() + 1e-12)
    d["range_z"] = (rng - rng.rolling(50).mean()) / (rng.rolling(50).std() + 1e-12)
    
    d["roc10"] = d["close"].pct_change(10)
    
    tp = (d["high"] + d["low"] + d["close"]) / 3.0
    mf = tp * d["volume"]
    pos_mf = mf.where(tp > tp.shift(1), 0.0)
    neg_mf = mf.where(tp < tp.shift(1), 0.0)
    mfr = pos_mf.rolling(14).sum() / (neg_mf.rolling(14).sum().abs() + 1e-12)
    d["mfi14"] = 100.0 - (100.0 / (1.0 + mfr))
    
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
        return labels # 1=Trend
    return 1 - labels

# ------------------------------
# 3. Labeling Targets (Range Only)
# ------------------------------
def label_targets(df: pd.DataFrame) -> pd.DataFrame:
    lookback = 12
    df["pivot_low"] = df["low"].rolling(window=lookback*2+1, center=True).min() == df["low"]
    # Task A: Range Reversal (Buy at Pivot Low in Range)
    df["target_range_buy"] = (df["pivot_low"] & (df["regime"] == 0)).astype(int)
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

pset = gp.PrimitiveSet("MAIN", 14)
pset.addPrimitive(add, 2, name="add")
pset.addPrimitive(sub, 2, name="sub")
pset.addPrimitive(mul, 2, name="mul")
pset.addPrimitive(protectedDiv, 2, name="div")
pset.addPrimitive(neg, 1, name="neg")
pset.addPrimitive(absv, 1, name="abs")
pset.addPrimitive(if_then, 3, name="if_gt_0")

arg_names = ["Close", "Open", "High", "Low", "Volume", "RSI", "SMA50", "EMA20", "ATR14", "BBWidth", "ROC10", "MFI14", "ADX", "RangeZ"]
for i, n in enumerate(arg_names): pset.renameArguments(**{f"ARG{i}": n})

if not hasattr(creator, "FitnessMax"): creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"): creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

GLOBAL_INPUTS = {}
GLOBAL_TARGET = None

def eval_formula(individual):
    func = toolbox.compile(expr=individual)
    try:
        args = [GLOBAL_INPUTS[k] for k in arg_names]
        output = func(*args)
        signal = (output > 0).astype(XP.int32)
        
        hits = _to_py_scalar(XP.sum((signal == 1) & (GLOBAL_TARGET == 1)))
        total_signals = _to_py_scalar(XP.sum(signal == 1))
        
        if total_signals == 0: return (0.0,)
        
        precision = float(hits) / float(total_signals)
        if total_signals < 10: precision *= 0.1
        
        target_ones = _to_py_scalar(XP.sum(GLOBAL_TARGET == 1))
        recall = float(hits) / float(target_ones + 1e-9)
        f1 = 2.0 * (precision * recall) / (precision + recall + 1e-9)
        return (float(f1),)
    except: return (0.0,)

toolbox.register("evaluate", eval_formula)

def algorithms_eaSimple(pop, toolbox, cxpb, mutpb, ngen, stats, halloffame, verbose, patience, min_delta):
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])
    
    # Evaluate Initial
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

def main():
    global XP
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", type=str, default="BTCUSDT")
    p.add_argument("--interval", type=str, default="15m")
    p.add_argument("--max_gens", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--min_delta", type=float, default=1e-4)
    p.add_argument("--pop_size", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_height", type=int, default=17)
    p.add_argument("--max_nodes", type=int, default=200)
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
    
    # Limits
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=args.max_height))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=args.max_height))
    
    # Load Data
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
    
    df = add_features(df)
    df["regime"] = classify_regimes(df)
    df = label_targets(df)
    
    global GLOBAL_INPUTS, GLOBAL_TARGET
    GLOBAL_INPUTS = {k: _maybe_to_xp(df[k.lower() if k not in ["BBWidth", "RangeZ"] else "bb_width" if k=="BBWidth" else "range_z"].values, args.use_gpu) for k in arg_names}
    GLOBAL_TARGET = _maybe_to_xp(df["target_range_buy"].values.astype(np.int32), args.use_gpu, XP.int32)
    
    print(f"--- Evolving Range Reversal (Regime=0) ---")
    pop = toolbox.population(n=args.pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    
    pop, log = algorithms_eaSimple(pop, toolbox, 0.5, 0.2, args.max_gens, stats, hof, True, args.patience, args.min_delta)
    
    best = hof[0]
    print(f"\n[Range Hunter] Best Formula: {best}")
    print(f"[Range Hunter] Best F1: {best.fitness.values[0]:.6f}")
    
    out_dir = f"./all_models/models_v20_1/{args.symbol}"
    _safe_mkdir(out_dir)
    with open(os.path.join(out_dir, "range_report.txt"), "w") as f:
        f.write(f"Task: Range Reversal\nFormula: {best}\nScore: {best.fitness.values[0]:.6f}\n")

if __name__ == "__main__":
    main()
