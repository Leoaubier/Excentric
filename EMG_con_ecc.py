import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


PUISSANCE = "40"

FIRST_FRAME_PLOT = 2000
END_FRAME_PLOT = 6000

# ============================================================
# Cycle detection
# ============================================================
def detect_cycles_from_q(q_ref, distance=100, prominence=None):
    """
    q_ref : signal 1D (ex: q[14,:])
    distance : distance minimale entre pics (en frames)
    """
    if prominence is None:
        prominence = 0.2 * np.std(q_ref)

    peaks, _ = find_peaks(
        q_ref,
        distance=distance
    )

    if len(peaks) < 2:
        raise RuntimeError("Pas assez de cycles détectés.")

    return peaks


# ============================================================
# EMG cycle normalization
# ============================================================
def normalize_emg_cycles(emg, peaks, n_points=200):
    """
    emg    : (n_muscles, n_frames)
    peaks  : indices de cycles
    return : (n_muscles, n_cycles, n_points)
    """
    n_muscles = emg.shape[0]
    cycles = []

    for i in range(len(peaks) - 1):
        i0, i1 = peaks[i], peaks[i+1]
        if i1 - i0 < 10:   # sécurité
            continue

        seg = emg[:, i0:i1]
        x_old = np.linspace(0, 1, seg.shape[1])
        x_new = np.linspace(0, 1, n_points)

        seg_norm = np.zeros((n_muscles, n_points))
        for m in range(n_muscles):
            seg_norm[m] = np.interp(x_new, x_old, seg[m])

        cycles.append(seg_norm)

    if len(cycles) == 0:
        raise RuntimeError("Aucun cycle valide.")

    cycles = np.stack(cycles, axis=1)  # (muscle, cycle, time)
    return cycles


# ============================================================
# Compute mean/std per mode
# ============================================================
def compute_mode_stats(q, emg, q_index=14, distance=200, n_points=200):
    q_ref = q[q_index, :]

    peaks = detect_cycles_from_q(
        q_ref,
        distance=distance
    )

    cycles = normalize_emg_cycles(
        emg,
        peaks,
        n_points=n_points
    )

    mean = np.mean(cycles, axis=1)
    std  = np.std(cycles, axis=1)

    return mean, std, cycles, peaks


# ============================================================
# LOAD DATA  (à adapter)
# ============================================================

# Concentrique
q_con   = np.load(f"/Users/leo/Desktop/Projet/Collecte_25_11/concentric_{PUISSANCE}W/q_inverse_kinematic.npy")[:,FIRST_FRAME_PLOT:END_FRAME_PLOT]
emg_con = np.load(f"/Users/leo/Desktop/Projet/Collecte_25_11/concentric_{PUISSANCE}W/emg_processed_resampled.npy")[:,FIRST_FRAME_PLOT:END_FRAME_PLOT]

# Excentrique (mets tes vrais chemins)
q_ecc   = np.load(f"/Users/leo/Desktop/Projet/Collecte_25_11/eccentric_{PUISSANCE}W/q_inverse_kinematic.npy")[:,FIRST_FRAME_PLOT:END_FRAME_PLOT]
emg_ecc = np.load(f"/Users/leo/Desktop/Projet/Collecte_25_11/eccentric_{PUISSANCE}W/emg_processed_resampled.npy")[:,FIRST_FRAME_PLOT:END_FRAME_PLOT]

assert q_con.shape[1] == emg_con.shape[1]
assert q_ecc.shape[1] == emg_ecc.shape[1]

n_muscles = emg_con.shape[0]
muscle_names = [
    "delt_ant",
    "triceps",
    "biceps",
    "trap_med",
    "delt_med",
    "trap_inf",
    "gd",
    "pec",
    "delt_post",
    "brachio",
    "trap_sup"
]


# ============================================================
# PARAMETERS
# ============================================================
DISTANCE = 100     # <-- à ajuster selon cadence (frames / cycle)
N_POINTS = 200

# ============================================================
# COMPUTE STATS
# ============================================================
mean_con, std_con, cycles_con, peaks_con = compute_mode_stats(
    q_con, emg_con, distance=DISTANCE, n_points=N_POINTS
)

mean_ecc, std_ecc, cycles_ecc, peaks_ecc = compute_mode_stats(
    q_ecc, emg_ecc, distance=DISTANCE, n_points=N_POINTS
)

print(f"Concentrique: {cycles_con.shape[1]} cycles")
print(f"Excentrique : {cycles_ecc.shape[1]} cycles")


qref = q_con[14, :]
plt.figure(figsize=(12,3))
plt.plot(qref, label="q[14,:]")
plt.plot(peaks_con, qref[peaks_con], "ro", label="peaks")
plt.legend()
plt.title(f"q[14,:] + peaks détectés (N={len(peaks_con)})")
plt.show()

qref = q_ecc[14, :]
plt.figure(figsize=(12,3))
plt.plot(qref, label="q[14,:]")
plt.plot(peaks_ecc, qref[peaks_ecc], "ro", label="peaks")
plt.legend()
plt.title(f"q[14,:] + peaks détectés (N={len(peaks_ecc)})")
plt.show()

print("peaks indices:", peaks_con[:20], "...")
print("diff(peaks) median:", np.median(np.diff(peaks_con)) if len(peaks_con)>1 else None)
# ============================================================
# PLOT
# ============================================================
x = np.linspace(0, 100, N_POINTS)

ncols = 3
nrows = int(np.ceil(n_muscles / ncols))
fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(15, 4*nrows),
    sharex=True,
)
axes = axes.flatten()



for m in range(n_muscles):
    ax = axes[m]

    ax.plot(x, mean_con[m], label=f"Concentrique")
    ax.fill_between(
        x,
        mean_con[m] - std_con[m],
        mean_con[m] + std_con[m],
        alpha=0.3
    )

    ax.plot(x, mean_ecc[m], label="Excentrique")
    ax.fill_between(
        x,
        mean_ecc[m] - std_ecc[m],
        mean_ecc[m] + std_ecc[m],
        alpha=0.3
    )

    ax.set_title(muscle_names[m])
    ax.set_xlabel("% cycle")
    ax.set_ylabel("EMG")
    ax.grid(True)

# remove empty subplots
for k in range(n_muscles, len(axes)):
    fig.delaxes(axes[k])

handles, labels = axes[0].get_legend_handles_labels()
# Marges compactes (titre proche de la figure)
fig.subplots_adjust(
    top=0.92,
    bottom=0.08,
    hspace=0.35,
    wspace=0.25
)

# Légende dans le coin bas droit
fig.legend(
    handles, labels,
    loc="lower right",
    bbox_to_anchor=(0.98, 0.02),
    frameon=True,
    fontsize=11
)

# Titre compact
fig.suptitle(
    f"EMG – Concentrique vs Excentrique (cycles via flexion du coude) à {PUISSANCE}W",
    fontsize=14,
    y=0.96
)

plt.show()