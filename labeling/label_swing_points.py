import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def identify_swing_points(df, window=5):
    """
    Identify swing highs and lows based on a local window.
    A candle is a Swing Low if it is lower than 'window' candles before and after.
    A candle is a Swing High if it is higher than 'window' candles before and after.
    """
    df['is_swing_low'] = False
    df['is_swing_high'] = False
    
    # We need look-ahead bias for labeling (identifying historical truth), 
    # but strictly forbidden for feature engineering.
    # rolling(window*2+1).min() centered
    
    # Efficient vectorized approach
    # Shift to align comparison
    # Low: low[i] < low[i-j] and low[i] < low[i+j] for all j in 1..window
    
    # Using rolling min/max with center=True
    # A point is a local min if it equals the rolling min over the window
    # Window size = 2 * window + 1
    roll_window = 2 * window + 1
    
    min_vals = df['low'].rolling(window=roll_window, center=True).min()
    max_vals = df['high'].rolling(window=roll_window, center=True).max()
    
    df['is_swing_low'] = (df['low'] == min_vals)
    df['is_swing_high'] = (df['high'] == max_vals)
    
    return df

def plot_swings(df, symbol, n_candles=300):
    """
    Plot the last n_candles with swing points marked.
    """
    subset = df.tail(n_candles).copy().reset_index(drop=True)
    
    plt.figure(figsize=(16, 8))
    plt.plot(subset.index, subset['close'], label='Close Price', color='gray', alpha=0.5)
    
    # Plot Swing Lows (Green Triangles Up)
    lows = subset[subset['is_swing_low']]
    plt.scatter(lows.index, lows['low'], color='green', marker='^', s=100, label='Swing Low', zorder=5)
    
    # Plot Swing Highs (Red Triangles Down)
    highs = subset[subset['is_swing_high']]
    plt.scatter(highs.index, highs['high'], color='red', marker='v', s=100, label='Swing High', zorder=5)
    
    plt.title(f"{symbol} Swing Points (Window=5)", fontsize=16)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = "swing_points_chart.png"
    plt.savefig(output_path)
    print(f"Chart saved to {output_path}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--symbol", type=str, default="BTCUSDT")
    p.add_argument("--interval", type=str, default="15m")
    args = p.parse_args()
    
    print(f"Loading data for {args.symbol} {args.interval}...")
    from huggingface_hub import snapshot_download
    path = snapshot_download(repo_id="zongowo111/cpb-models", repo_type="dataset")
    csv_file = None
    for root, _, files in os.walk(path):
        for f in files:
            if f.endswith(".csv") and args.symbol in f and args.interval in f:
                csv_file = os.path.join(root, f)
                break
                
    if not csv_file:
        print("File not found!")
        return

    df = pd.read_csv(csv_file)
    # Ensure time sorted
    time_col = next(c for c in df.columns if "time" in c)
    df = df.sort_values(time_col).reset_index(drop=True)
    
    print("Labeling Swing Points...")
    # Window=5 means: It's the lowest point among 5 candles before and 5 candles after.
    # Total 11 candles span. A strict local extremum.
    df = identify_swing_points(df, window=5)
    
    low_count = df['is_swing_low'].sum()
    high_count = df['is_swing_high'].sum()
    
    print(f"Total Candles: {len(df)}")
    print(f"Swing Lows Identified: {low_count}")
    print(f"Swing Highs Identified: {high_count}")
    
    plot_swings(df, args.symbol)

if __name__ == "__main__":
    main()
