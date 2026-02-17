#!/usr/bin/env python3
"""Plot train/validation loss from cnp_training_losses.csv. Saves to run_dir/plots and run_dir/analysis."""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt

def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    run_dir = os.path.abspath(run_dir)
    loss_csv = os.path.join(run_dir, 'cnp_training_losses.csv')
    if not os.path.isfile(loss_csv):
        print(f"Not found: {loss_csv}")
        sys.exit(1)
    df = pd.read_csv(loss_csv)
    if 'Train Loss' not in df.columns or 'Validation Loss' not in df.columns:
        print("Expected columns 'Train Loss' and 'Validation Loss'")
        sys.exit(1)
    epochs = df.get('Epoch', range(1, len(df) + 1))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, df['Train Loss'], label='Train Loss', color='C0', linewidth=1.5)
    ax.plot(epochs, df['Validation Loss'], label='Validation Loss', color='C1', linewidth=1.5)
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.set_title('Train / Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    plt.tight_layout()
    for out_dir in [os.path.join(run_dir, 'plots'), os.path.join(run_dir, 'analysis')]:
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, 'train_val_loss.png')
        plt.savefig(path, dpi=150)
        print(f"Saved: {path}")
    plt.close()

if __name__ == '__main__':
    main()
