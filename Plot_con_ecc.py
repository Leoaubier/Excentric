import numpy as np
import biorbd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ============================================================
# Cycle detection (même base que toi)
# ============================================================
def detect_cycles_from_q(q_ref, distance=100, prominence=None):
    q_ref = np.asarray(q_ref, dtype=float)
    if prominence is None:
        prominence = 0.2 * np.std(q_ref)

    peaks, _ = find_peaks(q_ref, distance=distance, prominence=prominence)

    if len(peaks) < 2:
        raise RuntimeError("Pas assez de cycles détectés (len(peaks)<2).")

    return peaks


# ============================================================
# Normalisation par cycle (générique multi-signaux)
# ============================================================
def normalize_cycles(signals, peaks, n_points=200, min_len=15):
    """
    signals : (n_signals, n_frames)
    peaks   : indices pics (dans la fenêtre)
    return  : (n_signals, n_cycles, n_points)
    """
    signals = np.asarray(signals, dtype=float)
    n_signals, _ = signals.shape

    cycles = []
    for i in range(len(peaks) - 1):
        i0, i1 = int(peaks[i]), int(peaks[i + 1])
        if (i1 - i0) < min_len:
            continue

        seg = signals[:, i0:i1]  # (n_signals, seg_len)
        x_old = np.linspace(0, 1, seg.shape[1])
        x_new = np.linspace(0, 1, n_points)

        seg_norm = np.zeros((n_signals, n_points))
        for k in range(n_signals):
            seg_norm[k] = np.interp(x_new, x_old, seg[k])

        cycles.append(seg_norm)

    if len(cycles) == 0:
        raise RuntimeError("Aucun cycle valide après filtrage min_len.")

    return np.stack(cycles, axis=1)  # (n_signals, n_cycles, n_points)


def normalize_1d_cycles(q_ref, peaks, n_points=200, min_len=15):
    """q_ref : (n_frames,) -> (n_cycles, n_points)"""
    q_ref = np.asarray(q_ref, dtype=float).reshape(1, -1)
    cyc = normalize_cycles(q_ref, peaks, n_points=n_points, min_len=min_len)  # (1, n_cycles, n_points)
    return cyc[0]  # (n_cycles, n_points)


# ============================================================
# Stats par mode (activations + forces) avec mêmes cycles
# ============================================================
def compute_mode_cycle_stats(
    q, activations, forces,
    q_index_ref=14,
    distance=100,
    prominence=None,
    n_points=200,
    min_len=15,
):
    """
    q           : (nbQ, n_frames)
    activations : (nbMuscles, n_frames)
    forces      : (nbMuscles, n_frames)
    """
    q = np.asarray(q, dtype=float)
    activations = np.asarray(activations, dtype=float)
    forces = np.asarray(forces, dtype=float)

    if q.shape[1] != activations.shape[1] or q.shape[1] != forces.shape[1]:
        raise ValueError("q, activations, forces doivent avoir le même n_frames (axis=1).")
    if activations.shape != forces.shape:
        raise ValueError("activations et forces doivent avoir la même shape (nbMuscles, n_frames).")

    q_ref = q[q_index_ref, :]
    peaks = detect_cycles_from_q(q_ref, distance=distance, prominence=prominence)

    act_cycles = normalize_cycles(activations, peaks, n_points=n_points, min_len=min_len)
    frc_cycles = normalize_cycles(forces, peaks, n_points=n_points, min_len=min_len)

    stats = {
        "peaks": peaks,
        "q_cycles": normalize_1d_cycles(q_ref, peaks, n_points=n_points, min_len=min_len),
        "act_cycles": act_cycles,
        "frc_cycles": frc_cycles,
        "act_mean": act_cycles.mean(axis=1),
        "act_std":  act_cycles.std(axis=1),
        "frc_mean": frc_cycles.mean(axis=1),
        "frc_std":  frc_cycles.std(axis=1),
    }
    return stats


# ============================================================
# Plot helpers
# ============================================================
def plot_q_alignment(q_cycles_con, q_cycles_ecc, title):
    x = np.linspace(0, 100, q_cycles_con.shape[1])

    plt.figure(figsize=(12, 4))
    for c in q_cycles_con:
        plt.plot(x, c, alpha=0.15)
    for c in q_cycles_ecc:
        plt.plot(x, c, alpha=0.15)

    plt.plot(x, q_cycles_con.mean(axis=0), linewidth=2.5, label=f"con (moy, N={q_cycles_con.shape[0]})")
    plt.plot(x, q_cycles_ecc.mean(axis=0), linewidth=2.5, label=f"ecc (moy, N={q_cycles_ecc.shape[0]})")

    plt.title(title)
    plt.xlabel("% cycle")
    plt.ylabel("q_ref (a.u.)")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def plot_q_with_peaks(q_ref, peaks, title):
    plt.figure(figsize=(12, 3))
    plt.plot(q_ref, label="q_ref")
    plt.plot(peaks, q_ref[peaks], "ro", label="peaks")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_grid_mean_std(mean_con, std_con, mean_ecc, std_ecc, muscle_names, y_label, suptitle):
    n_muscles, n_points = mean_con.shape
    x = np.linspace(0, 100, n_points)

    ncols = 5
    nrows = int(np.ceil(n_muscles / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows), sharex=True)
    axes = axes.flatten()

    for m in range(n_muscles):
        ax = axes[m]

        ax.plot(x, mean_con[m], label="Concentrique")
        ax.fill_between(x, mean_con[m] - std_con[m], mean_con[m] + std_con[m], alpha=0.25)

        ax.plot(x, mean_ecc[m], label="Excentrique")
        ax.fill_between(x, mean_ecc[m] - std_ecc[m], mean_ecc[m] + std_ecc[m], alpha=0.25)

        ax.set_title(muscle_names[m] if muscle_names is not None else f"muscle_{m}")
        ax.set_xlabel("% cycle")
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)

    for k in range(n_muscles, len(axes)):
        fig.delaxes(axes[k])

    handles, labels = axes[0].get_legend_handles_labels()

    fig.subplots_adjust(top=0.92, bottom=0.08, hspace=0.35, wspace=0.25)

    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.98, 0.02),
               frameon=True, fontsize=11)

    fig.suptitle(suptitle, fontsize=14, y=0.96)
    plt.show()


# ============================================================
# MAIN (à brancher sur tes arrays)
# ============================================================
if __name__ == "__main__":
    # -------------------------
    # Inputs (tu remplaces par tes chargements)
    # -------------------------
    # q_con, act_con, force_con : (.., n_frames)
    # q_ecc, act_ecc, force_ecc : (.., n_frames)
    #
    # Example:
    # q_con = np.load("...")[:, FIRST:END]
    # act_con = np.load("...")[:, FIRST:END]
    # force_con = np.load("...")[:, FIRST:END]

    PUISSANCE = "40"
    FIRST_FRAME_PLOT = 3000 # --> bien mettre sur les valeurs de static opti V3
    END_FRAME_PLOT = 4000

    # --- à adapter à tes chemins ---
    model = biorbd.Model("/Users/leo/Desktop/Projet/modele_opensim/wu_bras_gauche_seth_left_Sidonie.bioMod")
    muscle_names = [model.muscleNames()[i].to_string() for i in range(int(model.nbMuscles()))]

    q_con   = np.load(f"/Users/leo/Desktop/Projet/Collecte_25_11/concentric_{PUISSANCE}W/q_inverse_kinematic.npy")[:, FIRST_FRAME_PLOT:END_FRAME_PLOT]
    act_con = np.load(f"/Users/leo/Desktop/Projet/Collecte_25_11/concentric_{PUISSANCE}W/muscle_activations_nonlinear.npy")[:, :]
    frc_con = np.load(f"/Users/leo/Desktop/Projet/Collecte_25_11/concentric_{PUISSANCE}W/muscles_forces.npy")[:, :]

    q_ecc   = np.load(f"/Users/leo/Desktop/Projet/Collecte_25_11/eccentric_{PUISSANCE}W/q_inverse_kinematic.npy")[:, FIRST_FRAME_PLOT:END_FRAME_PLOT]
    act_ecc = np.load(f"/Users/leo/Desktop/Projet/Collecte_25_11/eccentric_{PUISSANCE}W/muscle_activations_nonlinear.npy")[:, :]
    frc_ecc = np.load(f"/Users/leo/Desktop/Projet/Collecte_25_11/eccentric_{PUISSANCE}W/muscles_forces.npy")[:, :]

    # -------------------------
    # Params
    # -------------------------
    Q_INDEX_REF = 14
    DISTANCE = 100
    PROM = None
    N_POINTS = 200
    MIN_LEN = 15

    # -------------------------
    # Compute stats
    # -------------------------
    stats_con = compute_mode_cycle_stats(
        q_con, act_con, frc_con,
        q_index_ref=Q_INDEX_REF,
        distance=DISTANCE,
        prominence=PROM,
        n_points=N_POINTS,
        min_len=MIN_LEN,
    )

    stats_ecc = compute_mode_cycle_stats(
        q_ecc, act_ecc, frc_ecc,
        q_index_ref=Q_INDEX_REF,
        distance=DISTANCE,
        prominence=PROM,
        n_points=N_POINTS,
        min_len=MIN_LEN,
    )

    print(f"Concentrique: {stats_con['act_cycles'].shape[1]} cycles")
    print(f"Excentrique : {stats_ecc['act_cycles'].shape[1]} cycles")
    print("diff(peaks_con) median:", np.median(np.diff(stats_con["peaks"])) if len(stats_con["peaks"]) > 1 else None)
    print("diff(peaks_ecc) median:", np.median(np.diff(stats_ecc["peaks"])) if len(stats_ecc["peaks"]) > 1 else None)

    # -------------------------
    # Plots : alignement cycles q
    # -------------------------
    plot_q_alignment(
        stats_con["q_cycles"],
        stats_ecc["q_cycles"],
        title=f"Vérification alignement cycles (q[{Q_INDEX_REF}] normalisé 0–100%)"
    )

    plot_q_with_peaks(
        q_con[Q_INDEX_REF, :], stats_con["peaks"],
        title=f"Concentrique — q[{Q_INDEX_REF}] + peaks (N={len(stats_con['peaks'])})"
    )

    plot_q_with_peaks(
        q_ecc[Q_INDEX_REF, :], stats_ecc["peaks"],
        title=f"Excentrique — q[{Q_INDEX_REF}] + peaks (N={len(stats_ecc['peaks'])})"
    )

    # -------------------------
    # Plots : activations (mean ± std)
    # -------------------------
    plot_grid_mean_std(
        stats_con["act_mean"], stats_con["act_std"],
        stats_ecc["act_mean"], stats_ecc["act_std"],
        muscle_names=muscle_names,
        y_label="Activation",
        suptitle=f"Activations — Concentrique vs Excentrique (cycles via q[{Q_INDEX_REF}]) à {PUISSANCE}W"
    )

    # -------------------------
    # Plots : forces (mean ± std)
    # -------------------------
    plot_grid_mean_std(
        stats_con["frc_mean"], stats_con["frc_std"],
        stats_ecc["frc_mean"], stats_ecc["frc_std"],
        muscle_names=muscle_names,
        y_label="Force musculaire (N)",
        suptitle=f"Forces musculaires — Concentrique vs Excentrique (cycles via q[{Q_INDEX_REF}]) à {PUISSANCE}W"
    )