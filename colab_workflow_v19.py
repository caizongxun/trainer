#!/usr/bin/env python3
"""colab_workflow_v19.py

V19 "The Alpha Factory" (Evolutionary Regime-Specific Formula Discovery)

What’s new vs V18
- Adds Early Stopping ("run indefinitely until it can no longer improve") via:
  - --max_gens: hard upper bound to prevent infinite runtime
  - --patience: stop after N consecutive generations with no improvement
  - --min_delta: minimum improvement threshold to count as progress
- Keeps backward compatibility:
  - --generations still exists; if --max_gens is not provided, --generations is used.
- Adds --seed for reproducibility.
- Writes extra metadata (best generation, early-stop reason) into the report.

Objective
- Step 1: Train a GMM Regime Classifier (Trend vs Range).
- Step 2: Use Genetic Programming (DEAP) to EVOLVE formulas specifically for:
  - Task A: Range Reversals (Buy Low / Sell High in Range)
  - Task B: Trend Start Detection (Breakout)
  - Task C: Trend End Detection (Climax)
- Step 3: Run many generations with early stopping to find the "Best Formula" for each task.

Run on Colab
!pip install deap && curl -s https://raw.githubusercontent.com/caizongxun/trainer/main/colab_workflow_v19.py | python3 - \
  --symbol BTCUSDT --interval 15m --pop_size 2000 \
  --max_gens 10000 --patience 100 --min_delta 1e-4

Artifacts
- text : ./all_models/models_v19/{symbol}/alpha_factory_report.txt
"""

import os
import argparse
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from deap import base, creator, tools, gp


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
    avg_gain = gain.ewm(alpha=1 / 14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    d["rsi"] = 100.0 - (100.0 / (1.0 + rs))

    # ADX (for Regime)
    dmp = d["high"].diff()
    dmm = d["low"].diff()
    dmp[dmp < 0] = 0
    dmm[dmm > 0] = 0
    tr1 = d["high"] - d["low"]
    tr2 = (d["high"] - d["close"].shift()).abs()
    tr3 = (d["low"] - d["close"].shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    plus_di = 100 * (dmp.ewm(alpha=1 / 14).mean() / atr)
    minus_di = 100 * (dmm.abs().ewm(alpha=1 / 14).mean() / atr)
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
# 3. Labeling Targets (The "Ground Truth")
# ------------------------------

def label_targets(df):
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
# 4. Genetic Programming Engine
# ------------------------------

def protectedDiv(left, right):
    with np.errstate(divide="ignore", invalid="ignore"):
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

pset.renameArguments(ARG0="Close")
pset.renameArguments(ARG1="Open")
pset.renameArguments(ARG2="High")
pset.renameArguments(ARG3="Low")
pset.renameArguments(ARG4="Volume")
pset.renameArguments(ARG5="RSI")
pset.renameArguments(ARG6="SMA50")

# Creator is global and cannot be re-created in same process.
# Safeguard for environments where user re-runs the cell.
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

        # Hits = Signal=1 AND Target=1
        hits = np.sum((signal == 1) & (GLOBAL_TARGET == 1))
        total_signals = np.sum(signal == 1)

        if total_signals == 0:
            return (0,)

        precision = hits / total_signals

        # Penalize if too few signals (we need at least 10 signals)
        if total_signals < 10:
            precision *= 0.1

        recall = hits / (np.sum(GLOBAL_TARGET == 1) + 1e-9)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-9)

        return (f1,)  # Optimize F1 Score
    except Exception:
        return (0,)


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
    """Minimal eaSimple variant with optional early stopping.

    Early stopping rule:
    - Track the best `stats['max']` value.
    - Stop when it has not improved by at least `min_delta` for `patience` consecutive generations.
    """
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    # Evaluate initial population
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

        # Crossover
        for i in range(1, len(offspring), 2):
            if random.random() < cxpb:
                toolbox.mate(offspring[i - 1], offspring[i])
                del offspring[i - 1].fitness.values
                del offspring[i].fitness.values

        # Mutation
        for i in range(len(offspring)):
            if random.random() < mutpb:
                toolbox.mutate(offspring[i])
                del offspring[i].fitness.values

        # Evaluate
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

    # Attach metadata for downstream reporting
    logbook.best_score = best_so_far
    logbook.best_gen = best_gen
    logbook.stopped_early = stopped_early
    logbook.stop_reason = stop_reason

    return population, logbook


def evolve_for_task(task_name, target_array, inputs, pop_size=50, gens=5, patience=0, min_delta=0.0):
    print(f"--- Evolving for Task: {task_name} ---")
    global GLOBAL_INPUTS, GLOBAL_TARGET
    GLOBAL_INPUTS = inputs
    GLOBAL_TARGET = target_array

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
    if getattr(log, "stopped_early", False):
        print(f"Stopped: EARLY (best_gen={log.best_gen}, best={log.best_score:.6f})")
    else:
        print(f"Stopped: REACHED_MAX_GENS (gens={gens}, best_gen={log.best_gen}, best={log.best_score:.6f})")

    meta = {
        "best_gen": int(getattr(log, "best_gen", 0)),
        "best_score": float(getattr(log, "best_score", best_score)),
        "stopped_early": bool(getattr(log, "stopped_early", False)),
        "stop_reason": str(getattr(log, "stop_reason", "")),
        "gens_limit": int(gens),
        "patience": int(patience),
        "min_delta": float(min_delta),
    }

    return str(best), best_score, meta


# ------------------------------
# Main
# ------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", type=str, default="BTCUSDT")
    p.add_argument("--interval", type=str, default="15m")

    # Backward compatible param (V18 style). Used when --max_gens is not provided.
    p.add_argument("--generations", type=int, default=10)

    # New (V19): "run indefinitely" with a safe upper bound + early stopping.
    p.add_argument("--max_gens", type=int, default=None)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--min_delta", type=float, default=1e-6)

    p.add_argument("--pop_size", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)

    max_gens = args.max_gens if args.max_gens is not None else args.generations

    print("[1/5] Loading Data...")
    from huggingface_hub import snapshot_download

    path = snapshot_download(
        repo_id="zongowo111/cpb-models",
        repo_type="dataset",
        allow_patterns=None,
        ignore_patterns=None,
    )
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

    # Prepare Inputs for GP
    inputs = {
        "Close": df["close"].values,
        "Open": df["open"].values,
        "High": df["high"].values,
        "Low": df["low"].values,
        "Volume": df["volume"].values,
        "RSI": df["rsi"].values,
        "SMA50": df["sma50"].values,
    }

    out_dir = f"./all_models/models_v19/{args.symbol}"
    _safe_mkdir(out_dir)
    report_file = os.path.join(out_dir, "alpha_factory_report.txt")

    with open(report_file, "w") as f:
        f.write(f"=== ALPHA FACTORY REPORT (V19) ({args.symbol}) ===\n\n")
        f.write(f"interval: {args.interval}\n")
        f.write(f"pop_size: {args.pop_size}\n")
        f.write(f"max_gens: {max_gens}\n")
        f.write(f"patience: {args.patience}\n")
        f.write(f"min_delta: {args.min_delta}\n")
        f.write(f"seed: {args.seed}\n\n")

        # Task A: Range Reversal
        best_formula, score, meta = evolve_for_task(
            "Range Reversal (Buy Low)",
            df["target_range_buy"].values,
            inputs,
            pop_size=args.pop_size,
            gens=max_gens,
            patience=args.patience,
            min_delta=args.min_delta,
        )
        f.write(
            "Task: Range Reversal\n"
            f"Formula: {best_formula}\n"
            f"Score (F1): {score:.6f}\n"
            f"best_gen: {meta['best_gen']}\n"
            f"stopped_early: {meta['stopped_early']}\n"
            f"stop_reason: {meta['stop_reason']}\n\n"
        )

        # Task B: Trend Start
        best_formula, score, meta = evolve_for_task(
            "Trend Start (Breakout)",
            df["target_trend_start"].values,
            inputs,
            pop_size=args.pop_size,
            gens=max_gens,
            patience=args.patience,
            min_delta=args.min_delta,
        )
        f.write(
            "Task: Trend Start\n"
            f"Formula: {best_formula}\n"
            f"Score (F1): {score:.6f}\n"
            f"best_gen: {meta['best_gen']}\n"
            f"stopped_early: {meta['stopped_early']}\n"
            f"stop_reason: {meta['stop_reason']}\n\n"
        )

        # Task C: Trend End
        best_formula, score, meta = evolve_for_task(
            "Trend End (Climax)",
            df["target_trend_end"].values,
            inputs,
            pop_size=args.pop_size,
            gens=max_gens,
            patience=args.patience,
            min_delta=args.min_delta,
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
