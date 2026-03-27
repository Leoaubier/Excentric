import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ESSAI = "Collecte_18_03"

PUISSANCE = "60"


# ============================================================
# 1️⃣ Rotation cohérente
# ============================================================
def ensure_forward_rotation(crank_angle, *signals):

    crank_angle = np.asarray(crank_angle, float)

    if np.median(np.diff(crank_angle)) < 0:
        crank_angle = crank_angle[::-1]
        signals = [s[..., ::-1] for s in signals]
        print("ECC inversé → remis dans le sens croissant")

    return (crank_angle, *signals)


# ============================================================
# 2️⃣ Détection cycles
# ============================================================
def detect_cycles_from_crank(crank_angle, min_cycle_frames=30):

    a = np.asarray(crank_angle)
    da = np.diff(a)
    wraps = np.where(da < -np.pi)[0] + 1

    valid = [wraps[0]]
    for s in wraps[1:]:
        if s - valid[-1] >= min_cycle_frames:
            valid.append(s)

    return np.array(valid)


# ============================================================
# 3️⃣ Normalisation par angle
# ============================================================
def normalize_cycles_by_crank(signals, crank_angle, starts, n_points=360):

    signals = np.asarray(signals)
    a = np.unwrap(crank_angle)

    angle_grid = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    cycles = []

    for i in range(len(starts)-1):

        i0, i1 = starts[i], starts[i+1]

        seg_a = a[i0:i1] - a[i0]
        seg_s = signals[:, i0:i1]

        phi = np.mod(seg_a, 2*np.pi)
        order = np.argsort(phi)

        phi = phi[order]
        seg_s = seg_s[:, order]

        interp = np.zeros((signals.shape[0], n_points))

        for m in range(signals.shape[0]):
            interp[m] = np.interp(angle_grid, phi, seg_s[m])

        cycles.append(interp)

    cycles = np.stack(cycles, axis=1)
    return cycles, angle_grid


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

# ============================================================
# 4️⃣ Pipeline stats
# ============================================================
def compute_emg_stats(emg, crank):

    starts = detect_cycles_from_crank(crank)

#    emg = emg[:,starts[0]:]
#    crank = crank[starts[0]:]

#    starts = starts-starts[0]

    cycles, angle_grid = normalize_cycles_by_crank(emg, crank, starts)

    plot_crank_with_starts(crank, starts, title=f"cycle")
    return {
        "mean": cycles.mean(axis=1),
        "std": cycles.std(axis=1),
        "angle_grid": angle_grid,
        "n_cycles": cycles.shape[1]
    }


# ============================================================
# 6️⃣ Polar helpers
# ============================================================
def segments_from_bool(mask):
    segs = []
    start = None
    for i, val in enumerate(mask):
        if val and start is None:
            start = i
        if not val and start is not None:
            segs.append((start, i))
            start = None
    if start is not None:
        segs.append((start, len(mask)))
    return segs


def merge_wrap_segments(segs, N):
    if len(segs) > 1 and segs[0][0] == 0 and segs[-1][1] == N:
        merged = [(segs[-1][0], segs[0][1] + N)]
        middle = segs[1:-1]
        return middle + merged
    return segs


def plot_polar_levels(ax, mean_profiles, muscle_names, angle_grid, title=""):

    LEVELS = [0.40, 0.60, 0.80]
    LW_MAP = {0.40: 2.0, 0.60: 5.0, 0.80: 8.0}

    m, N = mean_profiles.shape

    ax.set_theta_zero_location("N")
    ax.set_theta_direction(1)
    ax.set_yticks([])
    ax.set_ylim(-0.5, m + 1.0)
    ax.grid(True, alpha=0.25)
    ax.set_title(title, pad=18)

    ring_h = 0.75
    r0s = np.arange(m)

    cmap = plt.get_cmap("tab20")
    colors = [cmap(i % 20) for i in range(m)]

    for mi in range(m):

        prof = mean_profiles[mi]
        maxv = np.max(prof) if np.max(prof) > 0 else 1.0

        for level in sorted(LEVELS, reverse=True):

            thr = level * maxv
            mask = prof >= thr
            segs = merge_wrap_segments(segments_from_bool(mask), N)

            for s, e in segs:
                idx = np.arange(s, e)
                th = np.mod(angle_grid[idx % N], 2 * np.pi)
                r = np.ones_like(th) * (r0s[mi] + ring_h)

                ax.plot(th, r, linewidth=LW_MAP[level],
                        color=colors[mi], solid_capstyle="round")

    return colors


    # ============================================================
# ============================================================
# ============================ MAIN ===========================
# ============================================================
if __name__ == "__main__":

    if PUISSANCE == "40":
        START_CON = 2000  # frame de début (ex : 2000)
        END_CON = 5200  # frame de fin
        START_ECC = 5000  # frame de début (ex : 2000)
        END_ECC = 8000  # frame de fin
    elif PUISSANCE == "60":
        START_CON = 2000  # frame de début (ex : 2000)
        END_CON = 5000  # frame de fin
        START_ECC = 14000  # frame de début (ex : 2000)
        END_ECC = 17000  # frame de fin
    elif PUISSANCE == "80":
        START_CON = 1500  # frame de début (ex : 2000)
        END_CON = 4000  # frame de fin
        START_ECC = 7000  # frame de début (ex : 2000)
        END_ECC = 10000  # frame de fin
    else:
        print("PB PUISSANCE")

    emg_con = np.load(
        f"/Users/leo/Desktop/Projet/{ESSAI}/concentric_{PUISSANCE}W/emg_processed_resampled.npy"
    )[:, START_CON:END_CON]

    crank_con = np.load(
        f"/Users/leo/Desktop/Projet/{ESSAI}/concentric_{PUISSANCE}W/crank_angle.npy"
    )[START_CON:END_CON]

    emg_ecc = np.load(
        f"/Users/leo/Desktop/Projet/{ESSAI}/eccentric_{PUISSANCE}W/emg_processed_resampled.npy"
    )[:, START_ECC:END_ECC]

    crank_ecc = np.load(
        f"/Users/leo/Desktop/Projet/{ESSAI}/eccentric_{PUISSANCE}W/crank_angle.npy"
    )[START_ECC:END_ECC]

    crank_ecc, emg_ecc = ensure_forward_rotation(crank_ecc, emg_ecc)

    #crank_con, emg_con = set_common_angle_origin(crank_con, emg_con)
    #crank_ecc, emg_ecc = set_common_angle_origin(crank_ecc, emg_ecc)


    stats_con = compute_emg_stats(emg_con, crank_con)
    stats_ecc = compute_emg_stats(emg_ecc, crank_ecc)


    print("Cycles con :", stats_con["n_cycles"])
    print("Cycles ecc :", stats_ecc["n_cycles"])

    # ============================================================
    # Ordre anatomique
    # ============================================================

    muscle_names = [
        "delt_ant", "delt_med", "delt_post",
        "trap_sup", "triceps", "biceps",
        "trap_med", "trap_inf", "gd", "pec", "brachio"
    ]

    group_order = [
        ["delt_ant", "delt_med", "delt_post"],
        ["trap_inf", "trap_med", "trap_sup"],
        ["triceps", "biceps", "gd", "pec", "brachio"]
    ]

    ordered_indices = []
    for group in group_order:
        for name in group:
            ordered_indices.append(muscle_names.index(name))

    muscle_names_ordered = [muscle_names[i] for i in ordered_indices]

    # Réordonnage
    mean_con = stats_con["mean"][ordered_indices]
    std_con = stats_con["std"][ordered_indices]
    mean_ecc = stats_ecc["mean"][ordered_indices]
    std_ecc = stats_ecc["std"][ordered_indices]

    x_deg = np.rad2deg(stats_con["angle_grid"])
    ang_con = stats_con["angle_grid"]
    ang_ecc = stats_ecc["angle_grid"]

    # ============================================================
    # SUBPLOTS
    # ============================================================

    fig, axes = plt.subplots(4, 3, figsize=(12, 9), sharex=True)
    axes = axes.flatten()

    for i in range(len(ordered_indices)):
        ax = axes[i]

        ax.plot(x_deg, mean_con[i], label="Concentrique")
        ax.fill_between(x_deg, mean_con[i] - std_con[i],
                        mean_con[i] + std_con[i], alpha=0.25)

        ax.plot(x_deg, mean_ecc[i], label="Excentrique")
        ax.fill_between(x_deg, mean_ecc[i] - std_ecc[i],
                        mean_ecc[i] + std_ecc[i], alpha=0.25)

        ax.set_title(muscle_names[ordered_indices[i]])
        ax.set_xlim(0, 360)
        ax.grid(True, alpha=0.3)

    for j in range(len(ordered_indices), len(axes)):
        fig.delaxes(axes[j])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.98, 0.02),
               frameon=True, fontsize=11)
    fig.suptitle(f"EMG – Con vs Ecc ({PUISSANCE}W)")
    plt.tight_layout()
    plt.show()

    # ============================================================
    # POLAR PLOT
    # ============================================================

    fig = plt.figure(figsize=(14, 7))

    ax1 = fig.add_subplot(1, 2, 1, projection="polar")
    ax2 = fig.add_subplot(1, 2, 2, projection="polar")

    colors = plot_polar_levels(
        ax1, mean_con, muscle_names_ordered, ang_con,
        title=f"Concentrique {PUISSANCE}W"
    )

    plot_polar_levels(
        ax2, mean_ecc, muscle_names_ordered, ang_ecc,
        title=f"Excentrique {PUISSANCE}W"
    )

    # Légende niveaux
    LEVELS = [0.50, 0.70, 0.90]
    LW_MAP = {0.50: 2.0, 0.70: 5.0, 0.90: 8.0}

    legend_levels = [
        Line2D([0], [0], color="black", lw=LW_MAP[level],
               label=f"{int(level * 100)} % du max moyen")
        for level in LEVELS
    ]

    fig.legend(
        handles=legend_levels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=3,
        frameon=False
    )

    legend_muscles = [
        Line2D([0], [0], color=colors[i], lw=6,
               label=muscle_names_ordered[i])
        for i in range(len(muscle_names_ordered))
    ]

    fig.legend(
        handles=legend_muscles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
        frameon=False
    )

    plt.tight_layout(rect=[0, 0.08, 1, 0.92])
    plt.show()