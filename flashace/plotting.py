import matplotlib.pyplot as plt
import numpy as np
import os

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


def plot_training_metrics(history, save_dir="plots"):
    """
    Plot training curves for loss, energy, forces, and stresses versus epoch.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    epochs = np.arange(1, len(history.get("train_loss", [])) + 1)

    def _plot_series(y_train, y_val, ylabel, title, filename, logy=False):
        y_train = np.asarray(y_train, dtype=float)
        y_val = np.asarray(y_val, dtype=float)
        mask = np.isfinite(y_train) & np.isfinite(y_val)
        if not np.any(mask):
            return
        plt.figure(figsize=(8, 6))
        plt.plot(epochs[mask], y_train[mask], label="Train", linewidth=2)
        plt.plot(epochs[mask], y_val[mask], label="Validation", linewidth=2, linestyle="--")
        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14)
        if logy:
            plt.yscale("log")
        plt.legend()
        plt.grid(True, which="both", ls="-", alpha=0.5)
        plt.savefig(os.path.join(save_dir, filename), dpi=300)
        plt.close()

    if history.get("train_loss") and history.get("val_loss"):
        _plot_series(
            history["train_loss"],
            history["val_loss"],
            "Loss (Weighted)",
            "Training Convergence",
            "learning_curve.png",
            logy=True,
        )

    if history.get("train_energy_mev") and history.get("val_energy_mev"):
        _plot_series(
            history["train_energy_mev"],
            history["val_energy_mev"],
            "Energy RMSE (meV/atom)",
            "Energy vs Epoch",
            "energy_curve.png",
        )

    if history.get("train_force_rmse") and history.get("val_force_rmse"):
        _plot_series(
            history["train_force_rmse"],
            history["val_force_rmse"],
            "Force RMSE (eV/Å)",
            "Forces vs Epoch",
            "force_curve.png",
        )

    if history.get("train_stress_rmse") and history.get("val_stress_rmse"):
        _plot_series(
            history["train_stress_rmse"],
            history["val_stress_rmse"],
            "Stress RMSE",
            "Stresses vs Epoch",
            "stress_curve.png",
        )

    print(f"\n[PLOTTING] Metric plots saved to directory: '{save_dir}/'")
