import numpy as np
import biorbd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# ============================================================
# Cycle detection from crank angle (wrap 2pi)
# ============================================================
def detect_cycles_from_crank(crank_angle, min_cycle_frames=30):
    """
    Détecte les cycles par passage 2π -> 0 (wrap naturel).
    Fonctionne pour con et ecc automatiquement.
    """
    a = np.asarray(crank_angle, float)

    # différence brute
    da = np.diff(a)

    # seuil de wrap (plus robuste que -pi)
    threshold = np.pi

    # si rotation positive (concentrique)
    if np.median(da) > 0:
        wraps = np.where(da < -threshold)[0] + 1
    else:
        wraps = np.where(da > threshold)[0] + 1

    if wraps.size == 0:
        raise RuntimeError("Aucun wrap détecté dans crank_angle.")

    # filtrer cycles trop courts
    good = [wraps[0]]
    for s in wraps[1:]:
        if s - good[-1] >= min_cycle_frames:
            good.append(s)

    starts = np.array(good, dtype=int)

    if starts.size < 2:
        raise RuntimeError("Pas assez de cycles détectés.")

    return starts



# ============================================================
# Normalisation par angle (générique multi-signaux)
# ============================================================
def normalize_cycles_by_crank(signals, crank_angle, cycle_starts, n_points=360, min_samples=10):
    signals = np.asarray(signals, float)
    a = np.unwrap(np.asarray(crank_angle, float))

    n_signals, T = signals.shape
    angle_grid = np.linspace(0.0, 2*np.pi, n_points, endpoint=False)

    cycles = []
    for i in range(len(cycle_starts) - 1):
        i0 = int(cycle_starts[i])
        i1 = int(cycle_starts[i + 1])

        seg_s = signals[:, i0:i1]
        seg_a = a[i0:i1] - a[i0]     # peut être croissant OU décroissant

        # --- RECALAGE ANGULAIRE PHYSIQUE (mod 2π) ---
        seg_phi = np.mod(seg_a, 2*np.pi)   # [0, 2π)
        order = np.argsort(seg_phi)
        seg_phi = seg_phi[order]
        seg_s   = seg_s[:, order]

        # enlever doublons d'angle
        keep = np.concatenate(([True], np.diff(seg_phi) > 1e-9))
        seg_phi = seg_phi[keep]
        seg_s   = seg_s[:, keep]

        if seg_phi.size < min_samples:
            continue

        seg_norm = np.zeros((n_signals, n_points))
        for k in range(n_signals):
            seg_norm[k] = np.interp(angle_grid, seg_phi, seg_s[k])

        cycles.append(seg_norm)

    if len(cycles) == 0:
        raise RuntimeError("Aucun cycle valide (après filtrage).")

    cycles = np.stack(cycles, axis=1)  # (n_signals, n_cycles, n_points)
    return cycles, angle_grid



def normalize_1d_cycles_by_crank(sig_1d, crank_angle, cycle_starts, n_points=360):
    sig_1d = np.asarray(sig_1d, float).reshape(1, -1)
    cyc, angle_grid = normalize_cycles_by_crank(sig_1d, crank_angle, cycle_starts, n_points=n_points)
    return cyc[0], angle_grid  # (n_cycles, n_points), (n_points,)


# ============================================================
# Stats par mode (activations + forces) avec mêmes cycles (crank)
# ============================================================
def compute_mode_cycle_stats_crank(
    q, activations, forces, crank_angle,
    min_cycle_frames=30,
    n_points=360,
):
    """
    q           : (nbQ, n_frames)   (optionnel pour plots/diagnostic)
    activations : (nbMuscles, n_frames)
    forces      : (nbMuscles, n_frames)
    crank_angle : (n_frames,)
    """
    q = np.asarray(q, float)
    activations = np.asarray(activations, float)
    forces = np.asarray(forces, float)
    crank_angle = np.asarray(crank_angle, float)

    if q.shape[1] != activations.shape[1] or q.shape[1] != forces.shape[1] or q.shape[1] != crank_angle.shape[0]:
        raise ValueError("q, activations, forces, crank_angle doivent avoir le même n_frames.")
    if activations.shape != forces.shape:
        raise ValueError("activations et forces doivent avoir la même shape.")

    starts = detect_cycles_from_crank(crank_angle, min_cycle_frames=min_cycle_frames)

    act_cycles, angle_grid = normalize_cycles_by_crank(activations, crank_angle, starts, n_points=n_points)
    frc_cycles, _          = normalize_cycles_by_crank(forces,      crank_angle, starts, n_points=n_points)

    # (optionnel) normaliser aussi un q de réf pour vérifier l'alignement
    # ici on prend q[14] par défaut si dispo
    q_ref = q[14, :] if q.shape[0] > 14 else q[0, :]
    q_cycles, _ = normalize_1d_cycles_by_crank(q_ref, crank_angle, starts, n_points=n_points)

    stats = {
        "starts": starts,
        "angle_grid": angle_grid,
        "q_cycles": q_cycles,
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
def plot_q_alignment_angle(q_cycles_con, q_cycles_ecc, angle_grid, title):
    x_deg = (np.rad2deg(angle_grid) % 360)

    plt.figure(figsize=(12, 4))
    for c in q_cycles_con:
        plt.plot(x_deg, c, alpha=0.15)
    for c in q_cycles_ecc:
        plt.plot(x_deg, c, alpha=0.15)

    plt.plot(x_deg, q_cycles_con.mean(axis=0), linewidth=2.5, label=f"con (moy, N={q_cycles_con.shape[0]})")
    plt.plot(x_deg, q_cycles_ecc.mean(axis=0), linewidth=2.5, label=f"ecc (moy, N={q_cycles_ecc.shape[0]})")

    plt.title(title)
    plt.xlabel("Angle pédalier (deg)")
    plt.ylabel("q_ref (a.u.)")
    plt.xlim(0, 360)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def plot_crank_with_starts(crank_angle, starts, title):
    a = np.asarray(crank_angle, float)
    plt.figure(figsize=(12, 3))
    plt.plot(a, label="crank_angle (rad)")
    plt.plot(starts, a[starts], "ro", label="cycle starts")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_grid_mean_std_angle(mean_con, std_con, mean_ecc, std_ecc, muscle_names, angle_grid, y_label, suptitle):
    n_muscles, n_points = mean_con.shape
    x_deg = (np.rad2deg(angle_grid) % 360)

    ncols = 5
    nrows = int(np.ceil(n_muscles / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows), sharex=True)
    axes = np.atleast_1d(axes).flatten()

    for m in range(n_muscles):
        ax = axes[m]

        ax.plot(x_deg, mean_con[m], label="Concentrique")
        ax.fill_between(x_deg, mean_con[m] - std_con[m], mean_con[m] + std_con[m], alpha=0.25)

        ax.plot(x_deg, mean_ecc[m], label="Excentrique")
        ax.fill_between(x_deg, mean_ecc[m] - std_ecc[m], mean_ecc[m] + std_ecc[m], alpha=0.25)

        ax.set_title(muscle_names[m] if muscle_names is not None else f"muscle_{m}")
        ax.set_xlabel("Angle pédalier (deg)")
        ax.set_ylabel(y_label)
        ax.set_xlim(0, 360)
        ax.grid(True, alpha=0.3)

    # Supprimer axes vides
    for k in range(n_muscles, len(axes)):
        fig.delaxes(axes[k])

    # Légende globale
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
    PUISSANCE = "40"
    FIRST_FRAME_PLOT = 3000
    END_FRAME_PLOT = 4000
    n_frame = END_FRAME_PLOT - FIRST_FRAME_PLOT

    model = biorbd.Model("/Users/leo/Desktop/Projet/modele_opensim/wu_bras_gauche_seth_left_Sidonie_vtp.bioMod")
    muscle_names = [model.muscleNames()[i].to_string() for i in range(int(model.nbMuscles()))]

    # --- Concentrique ---
    q_con   = np.load(f"/Users/leo/Desktop/Projet/Collecte_25_11/concentric_{PUISSANCE}W/q_inverse_kinematic.npy")[:, FIRST_FRAME_PLOT:END_FRAME_PLOT]
    act_con = np.load(f"/Users/leo/Desktop/Projet/Collecte_25_11/concentric_{PUISSANCE}W/muscle_activations_nonlinear.npy")[:, :n_frame]
    frc_con = np.load(f"/Users/leo/Desktop/Projet/Collecte_25_11/concentric_{PUISSANCE}W/muscles_forces.npy")[:, :n_frame]
    crank_con = np.load(f"/Users/leo/Desktop/Projet/Collecte_25_11/concentric_{PUISSANCE}W/crank_angle.npy")[FIRST_FRAME_PLOT:END_FRAME_PLOT]

    # --- Excentrique ---
    q_ecc   = np.load(f"/Users/leo/Desktop/Projet/Collecte_25_11/eccentric_{PUISSANCE}W/q_inverse_kinematic.npy")[:, FIRST_FRAME_PLOT:END_FRAME_PLOT]
    act_ecc = np.load(f"/Users/leo/Desktop/Projet/Collecte_25_11/eccentric_{PUISSANCE}W/muscle_activations_nonlinear.npy")[:, :n_frame]
    frc_ecc = np.load(f"/Users/leo/Desktop/Projet/Collecte_25_11/eccentric_{PUISSANCE}W/muscles_forces.npy")[:, :n_frame]
    crank_ecc = np.load(f"/Users/leo/Desktop/Projet/Collecte_25_11/eccentric_{PUISSANCE}W/crank_angle.npy")[FIRST_FRAME_PLOT:END_FRAME_PLOT]

    # Checks
    assert q_con.shape[1] == crank_con.shape[0]
    assert q_ecc.shape[1] == crank_ecc.shape[0]
    assert act_con.shape[1] == crank_con.shape[0]
    assert act_ecc.shape[1] == crank_ecc.shape[0]

    # Params
    N_POINTS = 360
    MIN_CYCLE_FRAMES = 30

    # Compute stats (cycles via crank_angle)
    stats_con = compute_mode_cycle_stats_crank(
        q_con, act_con, frc_con, crank_con,
        min_cycle_frames=MIN_CYCLE_FRAMES,
        n_points=N_POINTS,
    )
    stats_ecc = compute_mode_cycle_stats_crank(
        q_ecc, act_ecc, frc_ecc, crank_ecc,
        min_cycle_frames=MIN_CYCLE_FRAMES,
        n_points=N_POINTS,
    )

    print(f"Concentrique: {stats_con['act_cycles'].shape[1]} cycles")
    print(f"Excentrique : {stats_ecc['act_cycles'].shape[1]} cycles")


    # Plots : check cycles
    plot_q_alignment_angle(
        stats_con["q_cycles"], stats_ecc["q_cycles"],
        stats_con["angle_grid"],
        title=f"Vérification alignement cycles (q_ref normalisé sur angle pédalier)"
    )

    plot_crank_with_starts(crank_con, stats_con["starts"], title=f"Concentrique — crank_angle + starts (N={len(stats_con['starts'])})")
    plot_crank_with_starts(crank_ecc, stats_ecc["starts"], title=f"Excentrique — crank_angle + starts (N={len(stats_ecc['starts'])})")

    # Activations
    plot_grid_mean_std_angle(
        stats_con["act_mean"], stats_con["act_std"],
        stats_ecc["act_mean"], stats_ecc["act_std"],
        muscle_names=muscle_names,
        angle_grid=stats_con["angle_grid"],
        y_label="Activation",
        suptitle=f"Activations — Concentrique vs Excentrique (abscisse = angle pédalier) à {PUISSANCE}W"
    )

    # Forces
    plot_grid_mean_std_angle(
        stats_con["frc_mean"], stats_con["frc_std"],
        stats_ecc["frc_mean"], stats_ecc["frc_std"],
        muscle_names=muscle_names,
        angle_grid=stats_con["angle_grid"],
        y_label="Force musculaire (N)",
        suptitle=f"Forces musculaires — Concentrique vs Excentrique (abscisse = angle pédalier) à {PUISSANCE}W"
    )
