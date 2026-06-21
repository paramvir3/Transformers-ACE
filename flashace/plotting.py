import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_metric_history(history, save_dir="plots"):
    """Save train/validation convergence curves and their numerical history."""
    os.makedirs(save_dir, exist_ok=True)

    epochs = np.asarray(history["epoch"], dtype=int)
    panels = [
        ("loss", "Weighted total loss"),
        ("energy_rmse", "Energy RMSE (meV/atom)"),
        ("force_rmse", r"Force RMSE (eV/$\AA$)"),
        ("stress_rmse", r"Stress RMSE (eV/$\AA^3$)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    for axis, (metric, ylabel) in zip(axes.flat, panels):
        plotted = False
        for split, style in (("train", "-"), ("val", "--")):
            values = np.asarray(history[f"{split}_{metric}"], dtype=float)
            positive = values > 0.0
            if np.any(positive):
                axis.plot(
                    epochs[positive],
                    values[positive],
                    style,
                    linewidth=2,
                    label="Training" if split == "train" else "Validation",
                )
                plotted = True
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.set_title(ylabel)
        if plotted:
            axis.set_yscale("log")
            axis.legend()
        else:
            axis.text(0.5, 0.5, "No labeled data", ha="center", va="center", transform=axis.transAxes)
        axis.grid(True, which="both", linestyle=":", alpha=0.45)

    figure_path = os.path.join(save_dir, "training_curves.png")
    fig.savefig(figure_path, dpi=250, bbox_inches="tight")
    plt.close(fig)

    history_path = os.path.join(save_dir, "training_history.csv")
    fieldnames = list(history.keys())
    with open(history_path, "w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(fieldnames)
        writer.writerows(zip(*(history[name] for name in fieldnames)))

    print(f"[PLOTTING] Training curves saved to: {figure_path}")
    print(f"[PLOTTING] Metric history saved to: {history_path}")
    return figure_path, history_path

def plot_training_results(history, train_preds, val_preds, save_dir="plots"):
    """
    Generates 3 plots: Learning Curve, Energy Parity, Force Parity.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # ---------------------------------------------------------
    # 1. LEARNING CURVE (Loss vs Epoch)
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 6))
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Val Loss', linewidth=2, linestyle='--')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss (Weighted)', fontsize=12)
    plt.title('Training Convergence', fontsize=14)
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.savefig(f"{save_dir}/learning_curve.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 2. ENERGY PARITY (DFT vs Prediction)
    # ---------------------------------------------------------
    plt.figure(figsize=(6, 6))
    
    # Unpack data
    t_e_true = np.array(train_preds['E_true'])
    t_e_pred = np.array(train_preds['E_pred'])
    v_e_true = np.array(val_preds['E_true'])
    v_e_pred = np.array(val_preds['E_pred'])
    
    # Calculate Per-Atom Energy for better scaling comparison
    # (Assuming normalization happened in training, if not, we plot total)
    
    # Plot Train
    plt.scatter(t_e_true, t_e_pred, alpha=0.5, s=15, label='Train', color='tab:blue')
    # Plot Val
    plt.scatter(v_e_true, v_e_pred, alpha=0.6, s=15, label='Validation', color='tab:orange', marker='x')
    
    # Perfect fit line
    all_e = np.concatenate([t_e_true, v_e_true])
    vmin, vmax = all_e.min(), all_e.max()
    plt.plot([vmin, vmax], [vmin, vmax], 'k--', lw=1.5)
    
    plt.xlabel('DFT Energy (eV)', fontsize=12)
    plt.ylabel('Predicted Energy (eV)', fontsize=12)
    plt.title(f'Energy Parity', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{save_dir}/energy_parity.png", dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 3. FORCE PARITY
    # ---------------------------------------------------------
    plt.figure(figsize=(6, 6))
    
    t_f_true = np.concatenate(train_preds['F_true'])
    t_f_pred = np.concatenate(train_preds['F_pred'])
    v_f_true = np.concatenate(val_preds['F_true'])
    v_f_pred = np.concatenate(val_preds['F_pred'])
    
    # Downsample if too many points for plotting speed (optional)
    if len(t_f_true) > 10000:
        idx = np.random.choice(len(t_f_true), 10000, replace=False)
        t_f_true, t_f_pred = t_f_true[idx], t_f_pred[idx]
        
    plt.scatter(t_f_true, t_f_pred, alpha=0.3, s=5, label='Train', color='tab:blue')
    plt.scatter(v_f_true, v_f_pred, alpha=0.4, s=5, label='Validation', color='tab:orange')
    
    all_f = np.concatenate([t_f_true, v_f_true])
    fmin, fmax = all_f.min(), all_f.max()
    plt.plot([fmin, fmax], [fmin, fmax], 'k--', lw=1.5)
    
    plt.xlabel(r'DFT Forces (eV/$\AA$)', fontsize=12)
    plt.ylabel(r'Predicted Forces (eV/$\AA$)', fontsize=12)
    plt.title('Force Parity', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{save_dir}/force_parity.png", dpi=300)
    plt.close()
    
    print(f"\n[PLOTTING] Plots saved to directory: '{save_dir}/'")
