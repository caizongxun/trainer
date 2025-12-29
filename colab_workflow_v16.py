#!/usr/bin/env python3
"""colab_workflow_v16.py

V16 "The Genetic Architect" (Evolutionary Formula Discovery)

Objective:
- Answer the user's request: "Can AI reverse engineer/invent a NEW formula?"
- Use Genetic Programming (GP) via DEAP library.
- Evolve a mathematical expression tree to maximize Profit Factor.
- Input Terminals: Open, High, Low, Close, Volume, RSI, SMA50, BB_Upper, BB_Lower.
- Output: A math formula string, e.g., "add(sub(Close, SMA50), mul(RSI, 0.5))".

Method:
- Population: 50 random formulas.
- Evolution: 10 generations of crossover/mutation.
- Fitness: Simple vectorized backtest PnL / Profit Factor.

Run on Colab:
!curl -s https://raw.githubusercontent.com/caizongxun/trainer/main/colab_workflow_v16.py | python3 - \
  --symbol BTCUSDT --interval 15m --generations 5

Artifacts:
- text : ./all_models/models_v16/{symbol}/evolved_formula.txt (The best formula found)
- plot : ./all_models/models_v16/{symbol}/plots/evolution_history.png
"""

import os
import argparse
import operator
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deap import base, creator, tools, gp

def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

# ------------------------------
# 1. Data Prep
# ------------------------------
def get_data(symbol, interval):
    from huggingface_hub import snapshot_download
    path = snapshot_download(repo_id="zongowo111/cpb-models", repo_type="dataset", allow_patterns=None, ignore_patterns=None)
    csv_file = None
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith(".csv") and symbol in f and interval in f:
                csv_file = os.path.join(root, f)
                break
        if csv_file: break
    if not csv_file: raise ValueError("No CSV found")
    
    df = pd.read_csv(csv_file)
    df.columns = [c.strip().lower() for c in df.columns]
    
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
    return df

# ------------------------------
# 2. Indicator Library (Terminals)
# ------------------------------
def prep_terminals(df):
    # Calculate some basics to feed the GP
    df["sma20"] = df["close"].rolling(20).mean()
    df["sma50"] = df["close"].rolling(50).mean()
    
    # RSI
    delta = df["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    df["rsi"] = 100.0 - (100.0 / (1.0 + rs))
    
    # BB
    std = df["close"].rolling(20).std()
    df["bb_up"] = df["sma20"] + 2*std
    df["bb_lo"] = df["sma20"] - 2*std
    
    df = df.dropna().reset_index(drop=True)
    
    # Return dict of arrays for evaluation
    return {
        "Close": df["close"].values,
        "Open": df["open"].values,
        "High": df["high"].values,
        "Low": df["low"].values,
        "Volume": df["volume"].values,
        "RSI": df["rsi"].values,
        "SMA50": df["sma50"].values,
        "BB_Up": df["bb_up"].values,
        "BB_Lo": df["bb_lo"].values
    }, df["close"].values # Return Close for PnL calc

# ------------------------------
# 3. Genetic Programming Setup
# ------------------------------
# Function Set
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

pset = gp.PrimitiveSet("MAIN", 9) # 9 terminals
pset.addPrimitive(np.add, 2, name="add")
pset.addPrimitive(np.subtract, 2, name="sub")
pset.addPrimitive(np.multiply, 2, name="mul")
pset.addPrimitive(protectedDiv, 2, name="div")
pset.addPrimitive(np.negative, 1, name="neg")
pset.addPrimitive(np.abs, 1, name="abs")
pset.addPrimitive(np.maximum, 2, name="max")
pset.addPrimitive(np.minimum, 2, name="min")
pset.addPrimitive(if_then, 3, name="if_gt_0") # Ternary operator: if arg1 > 0 then arg2 else arg3

# Terminals
# Args mapping: 0:Close, 1:Open, 2:High, 3:Low, 4:Volume, 5:RSI, 6:SMA50, 7:BB_Up, 8:BB_Lo
pset.renameArguments(ARG0='Close')
pset.renameArguments(ARG1='Open')
pset.renameArguments(ARG2='High')
pset.renameArguments(ARG3='Low')
pset.renameArguments(ARG4='Volume')
pset.renameArguments(ARG5='RSI')
pset.renameArguments(ARG6='SMA50')
pset.renameArguments(ARG7='BB_Up')
pset.renameArguments(ARG8='BB_Lo')

# Fitness: Maximize Profit Factor
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

# Evaluator
GLOBAL_DATA = {}
GLOBAL_CLOSE = []

def evalSymb(individual):
    # Transform expression into a signal array
    # If output > 0 -> Buy (1)
    # If output < 0 -> Sell (-1)
    # If output = 0 -> Hold (0)
    func = toolbox.compile(expr=individual)
    
    try:
        # Pass all arrays as arguments
        args = [GLOBAL_DATA[k] for k in ["Close", "Open", "High", "Low", "Volume", "RSI", "SMA50", "BB_Up", "BB_Lo"]]
        signal_raw = func(*args)
        
        # Vectorized Backtest
        # Position = sign(signal_raw)
        pos = np.sign(signal_raw)
        
        # Calculate PnL
        # ret = pos * log_ret (shifted)
        # But simply: if pos[t] == 1, we get (close[t+1] - close[t])
        close = GLOBAL_CLOSE
        log_ret = np.diff(np.log(close)) # len = N-1
        # align pos: pos[0] decides trade for ret[0] (t to t+1)
        # pos needs to be truncated to N-1
        strategy_ret = pos[:-1] * log_ret
        
        # Profit Factor
        gains = strategy_ret[strategy_ret > 0].sum()
        losses = -strategy_ret[strategy_ret < 0].sum()
        
        if losses == 0:
            return (0,) # Avoid inf, penalize no trades
        
        pf = gains / losses
        
        # Penalize inactivity (if num_trades < 10, score=0)
        n_trades = np.count_nonzero(np.diff(pos))
        if n_trades < 10:
            return (0,)
            
        return (pf,)
        
    except Exception as e:
        return (0,)

toolbox.register("evaluate", evalSymb)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)
toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))
toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=17))

# ------------------------------
# Main
# ------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", type=str, default="BTCUSDT")
    p.add_argument("--interval", type=str, default="15m")
    p.add_argument("--generations", type=int, default=5)
    args = p.parse_args()
    
    print(f"[1/3] Loading Data for {args.symbol}...")
    df = get_data(args.symbol, args.interval)
    data_dict, close_arr = prep_terminals(df)
    
    global GLOBAL_DATA, GLOBAL_CLOSE
    GLOBAL_DATA = data_dict
    GLOBAL_CLOSE = close_arr
    
    print(f"[2/3] Evolution Started ({args.generations} generations)...")
    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(1)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)
    
    pop, log = algorithms_eaSimple(pop, toolbox, 0.5, 0.2, args.generations, stats=stats, halloffame=hof, verbose=True)
    
    print("\n[3/3] Discovery Complete")
    best = hof[0]
    print("Best Formula Found:")
    print(str(best))
    print(f"Profit Factor: {best.fitness.values[0]:.2f}")
    
    out_dir = f"./all_models/models_v16/{args.symbol}"
    _safe_mkdir(os.path.join(out_dir, "plots"))
    
    with open(os.path.join(out_dir, "evolved_formula.txt"), "w") as f:
        f.write(f"Symbol: {args.symbol}\n")
        f.write(f"Interval: {args.interval}\n")
        f.write(f"Best Formula (DEAP Format): {str(best)}\n")
        f.write(f"Fitness (Profit Factor): {best.fitness.values[0]:.4f}\n")
        f.write("\nNOTE: This formula is raw math. 'add(A,B)' means A+B. 'if_gt_0(C, A, B)' means 'if C>0 then A else B'.\n")

    # Plot Evolution
    max_fitness = log.select("max")
    avg_fitness = log.select("avg")
    plt.figure(figsize=(10,5))
    plt.plot(max_fitness, label="Best Fitness")
    plt.plot(avg_fitness, label="Avg Fitness")
    plt.xlabel("Generation")
    plt.ylabel("Profit Factor")
    plt.title("Formula Evolution History")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "plots", "evolution_history.png"))
    print(f"Saved artifacts to {out_dir}")

# Custom eaSimple to fix import issues in script
def algorithms_eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
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

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Variate the pool of individuals
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

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook

if __name__ == "__main__":
    main()
