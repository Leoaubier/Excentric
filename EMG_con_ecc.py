import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PUISSANCE = "40"
FIRST_FRAME_PLOT = 2000
END_FRAME_PLOT = 6000


def wrap_to_pi(theta_rad):
    """Ramène un angle (rad) dans [-pi, pi)."""
    theta = np.asarray(theta_rad, float)
    return (theta + np.pi) % (2*np.pi) - np.pi

def wrap_to_180(theta_deg):
    """Ramène un angle (deg) dans [-180, 180)."""
    theta = np.asarray(theta_deg, float)
    return (theta + 180.0) % 360.0 - 180.0


# ============================================================
# Cycle detection from crank angle (wrap 2pi -> 0)
# ============================================================
def detect_cycles_from_crank(crank_angle, min_cycle_frames=30):
    a = np.asarray(crank_angle, float)

    # unwrap
    au = np.unwrap(a)

    # détecter sens de rotation
    slope = np.median(np.diff(au))

    # si rotation négative (excentrique)
    if slope < 0:
        au = -au  # inversion

    # remettre début à 0
    au = au - au[0]

    # identifiant de tour
    cycle_id = np.floor(au / (2*np.pi)).astype(int)
    changes = np.where(np.diff(cycle_id) > 0)[0] + 1

    starts = np.concatenate(([0], changes))

    # filtrer cycles trop courts
    good = [starts[0]]
    for s in starts[1:]:
        if s - good[-1] >= min_cycle_frames:
            good.append(s)

    starts = np.array(good, dtype=int)

    if starts.size < 2:
        raise RuntimeError(
            f"Pas assez de cycles détectés (starts={starts.size})."
        )

    return starts




# ============================================================
# Normalize cycles by crank angle (0..2pi grid)
# ============================================================
def normalize_emg_cycles_by_crank(emg, crank_angle, cycle_starts, n_points=360):
    emg = np.asarray(emg, float)
    a = np.unwrap(np.asarray(crank_angle, float))

    # sens de rotation: on force la rampe croissante
    if np.median(np.diff(a)) < 0:
        a = -a

    a = a - a[0]  # départ à 0

    m, T = emg.shape
    angle_grid = np.linspace(0.0, 2 * np.pi, n_points, endpoint=False)

    # grille équivalente en [0, 2pi) pour interp
    grid_0_2pi = (angle_grid + 2*np.pi) % (2*np.pi)

    cycles = []

    for i in range(len(cycle_starts) - 1):
        i0 = int(cycle_starts[i])
        i1 = int(cycle_starts[i+1])

        seg_emg = emg[:, i0:i1]
        seg_a   = a[i0:i1] - a[i0]  # commence à 0

        # garder 1 tour
        mask = seg_a < 2*np.pi
        if np.sum(mask) < 10:
            continue
        seg_a = seg_a[mask]
        seg_emg = seg_emg[:, mask]

        # interp nécessite x croissant: seg_a est croissant
        seg_norm = np.zeros((m, n_points))
        for mi in range(m):
            seg_norm[mi] = np.interp(grid_0_2pi, seg_a, seg_emg[mi])

        cycles.append(seg_norm)

    if len(cycles) == 0:
        raise RuntimeError("Aucun cycle valide.")

    cycles = np.stack(cycles, axis=1)  # (m, n_cycles, n_points)
    return cycles, angle_grid




def compute_mode_stats_crank(emg, crank_angle, n_points=360):
    starts = detect_cycles_from_crank(crank_angle)
    cycles, angle_grid = normalize_emg_cycles_by_crank(emg, crank_angle, starts, n_points=n_points)
    mean = np.mean(cycles, axis=1)  # (m, n_points)
    std  = np.std(cycles, axis=1)
    return mean, std, cycles, angle_grid, starts


# ------------------------------------------------------------
# Utils: segments from mask with wrap merge (for circular arcs)
# ------------------------------------------------------------
def segments_from_bool(mask):
    mask = np.asarray(mask, dtype=bool)
    N = mask.size
    if N == 0:
        return []
    d = np.diff(mask.astype(int))
    starts = list(np.where(d == 1)[0] + 1)
    ends   = list(np.where(d == -1)[0] + 1)
    if mask[0]:
        starts = [0] + starts
    if mask[-1]:
        ends = ends + [N]
    return list(zip(starts, ends))

def merge_wrap_segments(segs, N):
    if not segs:
        return []
    segs = sorted(segs, key=lambda x: x[0])
    merged = []
    cs, ce = segs[0]
    for s, e in segs[1:]:
        if s <= ce:
            ce = max(ce, e)
        else:
            merged.append((cs, ce))
            cs, ce = s, e
    merged.append((cs, ce))

    if len(merged) >= 2 and merged[0][0] == 0 and merged[-1][1] == N:
        # wrap merge
        first_s, first_e = merged[0]
        last_s, last_e   = merged[-1]
        new_seg = (last_s, N + first_e)
        merged = merged[1:-1]
        merged.insert(0, new_seg)
    return merged


# ============================================================
# LOAD DATA  (à adapter) : emg + crank_angle
# ============================================================
# Concentrique
emg_con = np.load(f"/Users/leo/Desktop/Projet/Collecte_25_11/concentric_{PUISSANCE}W/emg_processed_resampled.npy")[:, FIRST_FRAME_PLOT:END_FRAME_PLOT]
crank_con = np.load(f"/Users/leo/Desktop/Projet/Collecte_25_11/concentric_{PUISSANCE}W/crank_angle.npy")[FIRST_FRAME_PLOT:END_FRAME_PLOT]

# Excentrique
emg_ecc = np.load(f"/Users/leo/Desktop/Projet/Collecte_25_11/eccentric_{PUISSANCE}W/emg_processed_resampled.npy")[:, FIRST_FRAME_PLOT:END_FRAME_PLOT]
crank_ecc = np.load(f"/Users/leo/Desktop/Projet/Collecte_25_11/eccentric_{PUISSANCE}W/crank_angle.npy")[FIRST_FRAME_PLOT:END_FRAME_PLOT]

au = np.unwrap(crank_ecc)
au = au - au[0]

print("raw max:", np.max(crank_ecc))
print("unwrap max:", np.max(au))
print("approx nb tours:", np.max(au)/(2*np.pi))
print("nb wraps (diff < -pi):", np.sum(np.diff(crank_ecc) < -np.pi))


assert emg_con.shape[1] == crank_con.shape[0]
assert emg_ecc.shape[1] == crank_ecc.shape[0]

muscle_names = [
    "delt_ant","delt_med","delt_post","trap_sup","triceps","biceps",
    "trap_med","trap_inf","gd","pec","brachio"
]
n_muscles = emg_con.shape[0]

# ============================================================
# COMPUTE STATS
# ============================================================
N_POINTS = 360  # 1 point par degré (pratique)
mean_con, std_con, cycles_con, angle_grid, starts_con = compute_mode_stats_crank(emg_con, crank_con, n_points=N_POINTS)
mean_ecc, std_ecc, cycles_ecc, angle_grid2, starts_ecc = compute_mode_stats_crank(emg_ecc, crank_ecc, n_points=N_POINTS)

print(f"Concentrique: {cycles_con.shape[1]} cycles")
print(f"Excentrique : {cycles_ecc.shape[1]} cycles")

# ============================================================
# SUBPLOTS: mean ± std (abscisse = angle pédalier)
# ============================================================
x_deg = np.rad2deg(angle_grid)
x_deg = x_deg % 360

ncols = 3
nrows = int(np.ceil(n_muscles / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4*nrows), sharex=True)
axes = axes.flatten()

for m in range(n_muscles):
    ax = axes[m]

    ax.plot(x_deg, mean_con[m], label="Concentrique")
    ax.fill_between(x_deg, mean_con[m]-std_con[m], mean_con[m]+std_con[m], alpha=0.25)

    ax.plot(x_deg, mean_ecc[m], label="Excentrique")
    ax.fill_between(x_deg, mean_ecc[m]-std_ecc[m], mean_ecc[m]+std_ecc[m], alpha=0.25)

    ax.set_title(muscle_names[m])
    ax.set_xlabel("Angle pédalier (deg)")
    ax.set_ylabel("EMG")
    ax.set_xlim(0, 360)
    ax.grid(True, alpha=0.3)

for k in range(n_muscles, len(axes)):
    fig.delaxes(axes[k])

handles, labels = axes[0].get_legend_handles_labels()
fig.subplots_adjust(top=0.92, bottom=0.08, hspace=0.35, wspace=0.25)
fig.legend(handles, labels, loc="lower right", bbox_to_anchor=(0.98, 0.02), frameon=True, fontsize=11)
fig.suptitle(f"EMG – Concentrique vs Excentrique (abscisse = angle pédalier) à {PUISSANCE}W", fontsize=14, y=0.96)

plt.show()


# ============================================================
# POLAR PLOT: 3 line widths for >= 5%, 10%, 20% of max(mean)
# ============================================================
LEVELS = [0.30, 0.40, 0.50]
LW_MAP = {0.30: 2.0, 0.40: 5.0, 0.50: 8.0}

def plot_polar_levels(ax, mean_profiles, muscle_names, angle_grid, title=""):
    """
    mean_profiles: (m, N) = mean EMG over cycles as function of crank angle
    For each muscle, draw arcs where mean >= level * max(mean) with line width per level.
    """
    m, N = mean_profiles.shape

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.set_yticks([])
    ax.set_ylim(-0.5, m + 1.0)
    ax.grid(True, alpha=0.25)
    ax.set_title(title, pad=18)

    ring_h = 0.75
    r0s = np.arange(m)

    cmap = plt.get_cmap("tab20" if m <= 20 else "hsv")
    colors = [cmap(i % cmap.N) for i in range(m)]

    for mi in range(m):
        col = colors[mi]
        prof = mean_profiles[mi]
        maxv = np.max(prof) if np.max(prof) > 0 else 1.0

        # plot from highest level to lowest so thick arcs are on top
        for level in sorted(LEVELS, reverse=True):
            thr = level * maxv
            mask = prof >= thr
            segs = merge_wrap_segments(segments_from_bool(mask), N)

            for s, e in segs:
                # handle wrap segment where e can exceed N
                idx = np.arange(s, e)
                th = angle_grid[idx % N]
                r  = np.ones_like(th) * (r0s[mi] + ring_h)

                ax.plot(th, r, linewidth=LW_MAP[level], color=col, solid_capstyle="round")

    return colors

fig = plt.figure(figsize=(14, 7))
ax1 = fig.add_subplot(1, 2, 1, projection="polar")
ax2 = fig.add_subplot(1, 2, 2, projection="polar")

colors = plot_polar_levels(
    ax1, mean_con, muscle_names, angle_grid,
    title="Concentrique"
)
plot_polar_levels(
    ax2, mean_ecc, muscle_names, angle_grid,
    title="Excentrique"
)

# ---------------------------
# LÉGENDE MUSCLES (en bas)
# ---------------------------
legend_muscles = [
    Line2D([0], [0], color=colors[i], lw=6, label=muscle_names[i])
    for i in range(len(muscle_names))
]

# ---------------------------
# LÉGENDE NIVEAUX (en haut)
# ---------------------------
legend_levels = [
    Line2D([0], [0], color="black", lw=LW_MAP[level],
           label=f"{int(level*100)} % du max moyen")
    for level in sorted(LEVELS)
]

# Légende niveaux en haut (centrée)
fig.legend(
    handles=legend_levels,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.98),
    ncol=len(LEVELS),
    frameon=False,
    fontsize=11
)

# Légende muscles en bas (centrée)
fig.legend(
    handles=legend_muscles,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.02),
    ncol=4,   # adapte selon nb muscles
    frameon=False,
    fontsize=11
)

# Marges propres pour laisser place aux légendes
plt.tight_layout(rect=[0, 0.08, 1, 0.92])

plt.show()
