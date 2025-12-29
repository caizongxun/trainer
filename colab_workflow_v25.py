#!/usr/bin/env python3
"""colab_workflow_v25.py

V25 "The Trend Hunter" (Simplified & Focused)

Goal: Verify if GP can fundamentally identify trend starts.
Strategy:
1. Simplify Architecture: Back to Single-Formula.
2. Sharpen Target: Trend Start = Future 20-bar return > 2% AND Drawdown < 1%.
3. Focused Features: Only the most robust trend indicators (RSI, ADX, MACD, PriceAction).

Run on Colab (GPU):
!pip install deap cupy-cuda12x && curl -s https://raw.githubusercontent.com/caizongxun/trainer/main/colab_workflow_v25.py | python3 - \
  --symbol BTCUSDT --interval 15m --pop_size 2000 \
  --max_gens 5000 --patience 100 --min_delta 1e-4 \
  --max_height 8 --max_nodes 100 \
  --use_gpu
"""

import os
import argparse
import random
import operator
import numpy as np
import pandas as pd
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
# 1. Feature Engineering (Trend Focus)
# ------------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    
    # Basic
    d["sma20"] = d["close"].rolling(20).mean()
    d["sma50"] = d["close"].rolling(50).mean()
    d["ema20"] = d["close"].ewm(span=20, adjust=False).mean()
    d["std20"] = d["close"].rolling(20).std()
    
    # RSI
    delta = d["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    rs = gain.ewm(alpha=1/14).mean() / (loss.ewm(alpha=1/14).mean() + 1e-12)
    d["rsi"] = 100.0 - (100.0 / (1.0 + rs))
    
    # ADX (Trend Strength)
    tr = pd.concat([d["high"]-d["low"], (d["high"]-d["close"].shift()).abs(), (d["low"]-d["close"].shift()).abs()], axis=1).max(axis=1)
    d["atr14"] = tr.rolling(14).mean()
    dmp = d["high"].diff().clip(lower=0)
    dmm = d["low"].diff().clip(upper=0).abs()
    plus_di = 100 * (dmp.ewm(alpha=1/14).mean() / (d["atr14"] + 1e-12))
    minus_di = 100 * (dmm.ewm(alpha=1/14).mean() / (d["atr14"] + 1e-12))
    dx = (abs(plus_di - minus_di) / (abs(plus_di + minus_di) + 1e-12)) * 100
    d["adx"] = dx.rolling(14).mean().fillna(0)
    
    # MACD (Momentum)
    ema12 = d["close"].ewm(span=12, adjust=False).mean()
    ema26 = d["close"].ewm(span=26, adjust=False).mean()
    d["macd"] = ema12 - ema26
    d["macd_signal"] = d["macd"].ewm(span=9, adjust=False).mean()
    d["macd_hist"] = d["macd"] - d["macd_signal"]
    
    # Price Action
    d["body_height"] = (d["close"] - d["open"]).abs()
    d["upper_shadow"] = d["high"] - d[["close", "open"]].max(axis=1)
    d["lower_shadow"] = d[["close", "open"]].min(axis=1) - d["low"]
    
    d["roc10"] = d["close"].pct_change(10)
    
    d = d.dropna().reset_index(drop=True)
    return d

# ------------------------------
# 2. Labeling Targets (Trend Start)
# ------------------------------
def label_targets(df: pd.DataFrame) -> pd.DataFrame:
    # Definition: Price rises > 2.5% in next 20 bars
    # AND doesn't drop more than 1% during that time (Low Drawdown)
    
    future_window = 20
    min_return = 0.025
    max_drawdown = 0.01
    
    targets = np.zeros(len(df), dtype=int)
    
    for i in range(len(df) - future_window):
        entry_price = df["close"].iloc[i]
        future_prices = df["close"].iloc[i+1 : i+1+future_window]
        
        max_price = future_prices.max()
        min_price = future_prices.min()
        
        ret = (max_price - entry_price) / entry_price
        dd = (entry_price - min_price) / entry_price
        
        if ret >= min_return and dd <= max_drawdown:
            targets[i] = 1
            
    df["target_trend_start"] = targets
    return df

# ------------------------------
# 3. GP Primitives
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

pset = gp.PrimitiveSet("MAIN", 12) # Reduced inputs
pset.addPrimitive(add, 2, name="add")
pset.addPrimitive(sub, 2, name="sub")
pset.addPrimitive(mul, 2, name="mul")
pset.addPrimitive(protectedDiv, 2, name="div")
pset.addPrimitive(neg, 1, name="neg")
pset.addPrimitive(absv, 1, name="abs")
pset.addPrimitive(if_then, 3, name="if_gt_0")

arg_names = [
    "Close", "Volume", "RSI", "ADX", "MACD_Hist", "ROC10", 
    "SMA50", "EMA20", "ATR14", 
    "BodyHeight", "UpperShadow", "LowerShadow"
]
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
    # Bloat control
    if len(individual) > 100: return (0.0,)
    
    func = toolbox.compile(expr=individual)
    try:
        args = [GLOBAL_INPUTS[k] for k in arg_names]
        output = func(*args)
        signal = (output > 0).astype(XP.int32)
        
        hits = _to_py_scalar(XP.sum((signal == 1) & (GLOBAL_TARGET == 1)))
        total_signals = _to_py_scalar(XP.sum(signal == 1))
        
        if total_signals == 0: return (0.001,) # Non-zero penalty to encourage trading
        
        precision = float(hits) / float(total_signals)
        
        # Penalize low frequency but not as harshly as V23
        if total_signals < 5: precision *= 0.5
        
        target_ones = _to_py_scalar(XP.sum(GLOBAL_TARGET == 1))
        recall = float(hits) / float(target_ones + 1e-9)
        
        # V25: Focus on F1 again to find ANY signal first
        f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
        return (float(f1),)
    except: return (0.0,)

toolbox.register("evaluate", eval_formula)

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
    print(f"\n--- Evolving for Task: {task_name} (Simplified V25) ---")
    global GLOBAL_INPUTS, GLOBAL_TARGET
    GLOBAL_INPUTS = inputs
    GLOBAL_TARGET = target_array
    
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    
    pop, log = algorithms_eaSimple(pop, toolbox, 0.5, 0.2, gens, stats, hof, True, patience, min_delta)
    
    # Handle empty HallOfFame (though unlikely with simplified logic)
    if len(hof) > 0:
        best = hof[0]
        best_score = float(best.fitness.values[0])
        print(f"Best Formula: {best}")
        print(f"Best F1-Score: {best_score:.6f}")
        return str(best), best_score, {}
    else:
        print("No valid individual found.")
        return "N/A", 0.0, {}

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
    p.add_argument("--max_height", type=int, default=8)
    p.add_argument("--max_nodes", type=int, default=100)
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
    
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=args.max_height))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=args.max_height))
    
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
    
    print("[2/4] Feature Engineering (Trend Focus)...")
    df = add_features(df)
    
    print("[3/4] Labeling Targets (Trend Start)...")
    df = label_targets(df)
    
    # Fix: Map CamelCase names to snake_case DataFrame columns explicitly
    mapping = {
        "BodyHeight": "body_height",
        "UpperShadow": "upper_shadow",
        "LowerShadow": "lower_shadow",
        "MACD_Hist": "macd_hist"
    }
    
    inputs = {}
    for k in arg_names:
        col_name = mapping.get(k, k.lower())
        inputs[k] = _maybe_to_xp(df[col_name].values, args.use_gpu)
    
    out_dir = f"./all_models/models_v25/{args.symbol}"
    _safe_mkdir(out_dir)
    report_file = os.path.join(out_dir, "trend_hunter_report.txt")
    
    with open(report_file, "w") as f:
        f.write(f"=== THE TREND HUNTER REPORT (V25) ({args.symbol}) ===\n")
        
        target = _maybe_to_xp(df["target_trend_start"].values.astype(np.int32), args.use_gpu, XP.int32)
        best, score, meta = evolve_for_task("Trend Start", target, inputs, args.pop_size, args.max_gens, args.patience, args.min_delta, args.use_gpu)
        f.write(f"Task: Trend Start\nFormula: {best}\nScore: {score:.6f}\n\n")
        
    print(f"\n[4/4] Done! Report saved to {report_file}")

if __name__ == "__main__":
    main()
