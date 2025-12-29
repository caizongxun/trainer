#!/usr/bin/env python3
"""colab_workflow_v23.py

V23 "The Sniper" (High Precision Optimization)

Goal: Target F1/Precision > 0.7 by using a Dual-Formula architecture.
Logic: Buy ONLY if (Signal_Formula > 0) AND (Filter_Formula > 0).
      - Signal_Formula: Finds potential setups.
      - Filter_Formula: Rejects false positives.

Changes:
- GP Individual is now a pair of trees (dual-tree).
- Fitness function heavily weights Precision.
- Strict bloat control to prevent syntax errors.

Run on Colab (GPU):
!pip install deap cupy-cuda12x && curl -s https://raw.githubusercontent.com/caizongxun/trainer/main/colab_workflow_v23.py | python3 - \
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
    
    # Price Action
    d["upper_shadow"] = d["high"] - d[["close", "open"]].max(axis=1)
    d["lower_shadow"] = d[["close", "open"]].min(axis=1) - d["low"]
    d["body_height"] = (d["close"] - d["open"]).abs()
    d["gap"] = d["open"] - d["close"].shift(1)
    
    d["hour"] = d["open_time"].dt.hour
    d["day_of_week"] = d["open_time"].dt.dayofweek
    
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
    "UpperShadow", "LowerShadow", "BodyHeight", "Gap", "Hour", "DayOfWeek"
]
for i, n in enumerate(arg_names): pset.renameArguments(**{f"ARG{i}": n})

# Multi-Tree Individual
if not hasattr(creator, "FitnessMax"): creator.create("FitnessMax", base.Fitness, weights=(1.0,))
if not hasattr(creator, "Individual"): creator.create("Individual", list, fitness=creator.FitnessMax) # list of trees

toolbox = base.Toolbox()
# Tree 1: Signal
toolbox.register("expr_signal", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
# Tree 2: Filter
toolbox.register("expr_filter", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)

def init_individual(icls, content):
    return icls(content)

toolbox.register("individual", tools.initCycle, creator.Individual, (lambda: gp.PrimitiveTree(toolbox.expr_signal()), lambda: gp.PrimitiveTree(toolbox.expr_filter())), n=1)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# Crossover for Dual Trees
def cxDualTree(ind1, ind2):
    # 50% chance swap Signal trees, 50% chance swap Filter trees
    if random.random() < 0.5:
        gp.cxOnePoint(ind1[0], ind2[0])
    else:
        gp.cxOnePoint(ind1[1], ind2[1])
    return ind1, ind2

# Mutation for Dual Trees
def mutDualTree(ind, expr):
    # 50% chance mutate Signal, 50% chance mutate Filter
    if random.random() < 0.5:
        gp.mutUniform(ind[0], expr, pset)
    else:
        gp.mutUniform(ind[1], expr, pset)
    return ind,

toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", cxDualTree)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", mutDualTree, expr=toolbox.expr_mut)

GLOBAL_INPUTS = {}
GLOBAL_TARGET = None

def eval_dual_formula(individual):
    # ind[0] = signal, ind[1] = filter
    # BLOAT CONTROL
    if len(individual[0]) > 100 or len(individual[1]) > 100:
        return (0.0,)
        
    func_sig = toolbox.compile(expr=individual[0])
    func_fil = toolbox.compile(expr=individual[1])
    
    try:
        args = [GLOBAL_INPUTS[k] for k in arg_names]
        out_sig = func_sig(*args)
        out_fil = func_fil(*args)
        
        # Dual Logic: Buy if Signal > 0 AND Filter > 0
        final_signal = ((out_sig > 0) & (out_fil > 0)).astype(XP.int32)
        
        hits = _to_py_scalar(XP.sum((final_signal == 1) & (GLOBAL_TARGET == 1)))
        total_signals = _to_py_scalar(XP.sum(final_signal == 1))
        
        if total_signals == 0: return (0.0,)
        
        precision = float(hits) / float(total_signals)
        
        # Heavily PENALIZE if signals < 10 (Avoid overfitting to 1-2 samples)
        if total_signals < 10: precision = 0.0
        
        target_ones = _to_py_scalar(XP.sum(GLOBAL_TARGET == 1))
        recall = float(hits) / float(target_ones + 1e-9)
        
        # V23 Goal: Precision matters MORE than Recall
        # Weighted F-Score (Beta=0.5 means precision is 2x more important)
        beta = 0.5
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
    print(f"\n--- Evolving for Task: {task_name} (Dual-Formula Sniper Mode) ---")
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
    p.add_argument("--max_height", type=int, default=6) # Strict
    p.add_argument("--max_nodes", type=int, default=80) # Strict
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
    
    # Static limits for dual trees? 
    # Decorating 'mate' for list-of-trees is tricky in standard DEAP.
    # We handle size limits inside 'eval_dual_formula' instead.
    
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
    
    print("[2/5] Feature Engineering (Price Action + Time)...")
    df = add_features(df)
    
    print("[3/5] Classifying Regimes (GMM)...")
    df["regime"] = classify_regimes(df)
    
    print("[4/5] Labeling Targets...")
    df = label_targets(df)
    
    inputs = {k: _maybe_to_xp(df[k.lower() if k not in ["BBWidth", "RangeZ", "UpperShadow", "LowerShadow", "BodyHeight", "Gap", "Hour", "DayOfWeek"] else 
                              "bb_width" if k=="BBWidth" else 
                              "range_z" if k=="RangeZ" else
                              "upper_shadow" if k=="UpperShadow" else
                              "lower_shadow" if k=="LowerShadow" else
                              "body_height" if k=="BodyHeight" else
                              "gap" if k=="Gap" else
                              "hour" if k=="Hour" else
                              "day_of_week"].values, args.use_gpu) for k in arg_names}
    
    out_dir = f"./all_models/models_v23/{args.symbol}"
    _safe_mkdir(out_dir)
    report_file = os.path.join(out_dir, "sniper_report.txt")
    
    with open(report_file, "w") as f:
        f.write(f"=== THE SNIPER REPORT (V23) ({args.symbol}) ===\n")
        f.write(f"Goal: High Precision via Dual-Formula (Signal + Filter)\n\n")
        
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
