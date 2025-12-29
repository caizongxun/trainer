#!/usr/bin/env python3
"""colab_workflow_v18.py

V18 "The Alpha Factory" (Evolutionary Regime-Specific Formula Discovery)

Objective:
- Fulfill the user's request for "creating hundreds/thousands of formulas" to find the best one for specific market states.
- Step 1: Train a GMM Regime Classifier (Trend vs Range).
- Step 2: Use Genetic Programming (DEAP) to EVOLVE formulas specifically for:
  - Task A: Range Reversals (Buy Low / Sell High in Range)
  - Task B: Trend Start Detection (Breakout)
  - Task C: Trend End Detection (Climax)
- Step 3: Run hundreds of generations to find the "Best Formula" for each task.

Run on Colab:
!pip install deap && curl -s https://raw.githubusercontent.com/caizongxun/trainer/main/colab_workflow_v18.py | python3 - \
  --symbol BTCUSDT --interval 15m --generations 10 --pop_size 100

Artifacts:
- text : ./all_models/models_v18/{symbol}/alpha_factory_report.txt
"""

import os
import argparse
import operator
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from deap import base, creator, tools, gp

def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

# ------------------------------
# 1. Feature Engineering
# ------------------------------
def add_features(df):
    d = df.copy()
    # Basic technicals for GP inputs
    d["sma20"] = d["close"].rolling(20).mean()
    d["sma50"] = d["close"].rolling(50).mean()
    d["std20"] = d["close"].rolling(20).std()
    
    # RSI
    delta = d["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    d["rsi"] = 100.0 - (100.0 / (1.0 + rs))
    
    # ADX (for Regime)
    dmp = d['high'].diff()
    dmm = d['low'].diff()
    dmp[dmp < 0] = 0
    dmm[dmm > 0] = 0
    tr1 = d['high'] - d['low']
    tr2 = (d['high'] - d['close'].shift()).abs()
    tr3 = (d['low'] - d['close'].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    plus_di = 100 * (dmp.ewm(alpha=1/14).mean() / atr)
    minus_di = 100 * (dmm.abs().ewm(alpha=1/14).mean() / atr)
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    d["adx"] = dx.rolling(14).mean().fillna(0)
    
    # Volatility Z
    d["range"] = (d["high"] - d["low"]) / d["close"]
    d["range_z"] = (d["range"] - d["range"].rolling(50).mean()) / d["range"].rolling(50).std()

    d = d.dropna().reset_index(drop=True)
    return d

# ------------------------------
# 2. Regime Classification (The "Temporary Model")
# ------------------------------
def classify_regimes(df):
    # Features for clustering
    X = df[["adx", "range_z"]].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # GMM with 2 components
    gmm = GaussianMixture(n_components=2, random_state=42)
    labels = gmm.fit_predict(X_scaled)
    
    # Identify which label is Trend (Higher ADX)
    mean_adx_0 = df.loc[labels==0, "adx"].mean()
    mean_adx_1 = df.loc[labels==1, "adx"].mean()
    
    if mean_adx_1 > mean_adx_0:
        return labels # 1=Trend, 0=Range
    else:
        return 1 - labels # Flip so 1=Trend

# ------------------------------
# 3. Labeling Targets (The "Ground Truth")
# ------------------------------
def label_targets(df):
    # We need specific targets for specific tasks
    
    # Pivot Points (Strict)
    lookback = 12
    df["pivot_low"] = df["low"].rolling(window=lookback*2+1, center=True).min() == df["low"]
    df["pivot_high"] = df["high"].rolling(window=lookback*2+1, center=True).max() == df["high"]
    
    # Task A: Range Reversal (Buy at Pivot Low in Range)
    df["target_range_buy"] = (df["pivot_low"] & (df["regime"] == 0)).astype(int)
    
    # Task B: Trend Start (Buy when Trend starts)
    # Define Trend Start as: Regime switches 0->1 AND Price goes up in next 10 bars
    df["regime_switch"] = (df["regime"] == 1) & (df["regime"].shift(1) == 0)
    df["future_ret"] = df["close"].shift(-10) / df["close"] - 1
    df["target_trend_start"] = (df["regime_switch"] & (df["future_ret"] > 0.02)).astype(int) # >2% pump
    
    # Task C: Trend End (Sell at Pivot High in Trend)
    df["target_trend_end"] = (df["pivot_high"] & (df["regime"] == 1)).astype(int)
    
    return df

# ------------------------------
# 4. Genetic Programming Engine
# ------------------------------
# GP Setup
def protectedDiv(left, right):
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.divide(left, right)
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            return 1
    return x

def if_then(condition, out1, out2):
    return np.where(condition > 0, out1, out2)

pset = gp.PrimitiveSet("MAIN", 7)
pset.addPrimitive(np.add, 2, name="add")
pset.addPrimitive(np.subtract, 2, name="sub")
pset.addPrimitive(np.multiply, 2, name="mul")
pset.addPrimitive(protectedDiv, 2, name="div")
pset.addPrimitive(np.negative, 1, name="neg")
pset.addPrimitive(np.abs, 1, name="abs")
pset.addPrimitive(if_then, 3, name="if_gt_0")

pset.renameArguments(ARG0='Close')
pset.renameArguments(ARG1='Open')
pset.renameArguments(ARG2='High')
pset.renameArguments(ARG3='Low')
pset.renameArguments(ARG4='Volume')
pset.renameArguments(ARG5='RSI')
pset.renameArguments(ARG6='SMA50')

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
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

# Global Data for Eval
GLOBAL_INPUTS = {}
GLOBAL_TARGET = []

def eval_formula(individual):
    func = toolbox.compile(expr=individual)
    try:
        args = [GLOBAL_INPUTS[k] for k in ["Close", "Open", "High", "Low", "Volume", "RSI", "SMA50"]]
        output = func(*args)
        
        # Convert output to signal (threshold > 0)
        signal = (output > 0).astype(int)
        
        # Calculate Precision (we want high precision for signals)
        # Hits = Signal=1 AND Target=1
        hits = np.sum((signal == 1) & (GLOBAL_TARGET == 1))
        total_signals = np.sum(signal == 1)
        
        if total_signals == 0: return (0,)
        
        precision = hits / total_signals
        
        # Penalize if too few signals (we need at least 10 signals)
        if total_signals < 10: precision *= 0.1
        
        # Reward recall slightly so it doesn't just pick 1 lucky point
        recall = hits / (np.sum(GLOBAL_TARGET == 1) + 1e-9)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
        
        return (f1,) # Optimize F1 Score
        
    except:
        return (0,)

toolbox.register("evaluate", eval_formula)

def evolve_for_task(task_name, target_array, inputs, pop_size=50, gens=5):
    print(f"--- Evolving for Task: {task_name} ---")
    global GLOBAL_INPUTS, GLOBAL_TARGET
    GLOBAL_INPUTS = inputs
    GLOBAL_TARGET = target_array
    
    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("max", np.max)
    
    pop, log = algorithms_eaSimple(pop, toolbox, 0.5, 0.2, gens, stats=stats, halloffame=hof, verbose=True)
    
    best = hof[0]
    print(f"Best Formula: {best}")
    print(f"Best Score: {best.fitness.values[0]:.4f}")
    return str(best), best.fitness.values[0]

# ------------------------------
# Main
# ------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", type=str, default="BTCUSDT")
    p.add_argument("--interval", type=str, default="15m")
    p.add_argument("--generations", type=int, default=10)
    p.add_argument("--pop_size", type=int, default=100)
    args = p.parse_args()

    print("[1/5] Loading Data...")
    from huggingface_hub import snapshot_download
    path = snapshot_download(repo_id="zongowo111/cpb-models", repo_type="dataset", allow_patterns=None, ignore_patterns=None)
    csv_file = None
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith(".csv") and args.symbol in f and args.interval in f:
                csv_file = os.path.join(root, f)
                break
        if csv_file: break
    df = pd.read_csv(csv_file)
    
    # Parse time
    time_col = next((c for c in ["open_time", "opentime", "timestamp", "time", "date"] if c in df.columns), None)
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
    print(f"Range Candles: {sum(df['regime']==0)}, Trend Candles: {sum(df['regime']==1)}")
    
    print("[4/5] Labeling Targets...")
    df = label_targets(df)
    
    # Prepare Inputs for GP
    inputs = {
        "Close": df["close"].values,
        "Open": df["open"].values,
        "High": df["high"].values,
        "Low": df["low"].values,
        "Volume": df["volume"].values,
        "RSI": df["rsi"].values,
        "SMA50": df["sma50"].values
    }
    
    # Evolution Tasks
    out_dir = f"./all_models/models_v18/{args.symbol}"
    _safe_mkdir(out_dir)
    report_file = os.path.join(out_dir, "alpha_factory_report.txt")
    
    with open(report_file, "w") as f:
        f.write(f"=== ALPHA FACTORY REPORT ({args.symbol}) ===\n\n")
        
        # Task A: Range Reversal
        best_formula, score = evolve_for_task("Range Reversal (Buy Low)", df["target_range_buy"].values, inputs, args.pop_size, args.generations)
        f.write(f"Task: Range Reversal\nFormula: {best_formula}\nScore (F1): {score:.4f}\n\n")
        
        # Task B: Trend Start
        best_formula, score = evolve_for_task("Trend Start (Breakout)", df["target_trend_start"].values, inputs, args.pop_size, args.generations)
        f.write(f"Task: Trend Start\nFormula: {best_formula}\nScore (F1): {score:.4f}\n\n")
        
        # Task C: Trend End
        best_formula, score = evolve_for_task("Trend End (Climax)", df["target_trend_end"].values, inputs, args.pop_size, args.generations)
        f.write(f"Task: Trend End\nFormula: {best_formula}\nScore (F1): {score:.4f}\n\n")

    print(f"\n[5/5] Done! Report saved to {report_file}")

# Custom eaSimple to fix import issues in script
def algorithms_eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

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

    return population, logbook

if __name__ == "__main__":
    main()
