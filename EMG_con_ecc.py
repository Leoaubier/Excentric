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

def normalize_q_cycles(q_ref, peaks, n_points=200, min_len=15):
    """
    q_ref : (n_frames_window,) -> déjà fenêtré si tu as FRAME_START/END
    peaks : indices dans cette fenêtre
    return: (n_cycles, n_points)
    """
    q_ref = np.asarray(q_ref, dtype=float)
    cycles = []

    for i in range(len(peaks) - 1):
        i0, i1 = int(peaks[i]), int(peaks[i+1])
        if (i1 - i0) < min_len:
            continue

        seg = q_ref[i0:i1]
        x_old = np.linspace(0, 1, seg.shape[0])
        x_new = np.linspace(0, 1, n_points)
        seg_norm = np.interp(x_new, x_old, seg)
        cycles.append(seg_norm)

    if len(cycles) == 0:
        raise RuntimeError("Aucun cycle Q valide pour le plot d'alignement.")

    return np.stack(cycles, axis=0)  # (n_cycles, n_points)

def pick_segment_containing_peak(segs, peak_idx, N):
    """
    segs: list[(s,e)] potentiellement "déroulés" (e peut dépasser N après merge_wrap_segments)
    peak_idx: int dans [0, N-1]
    """
    if not segs:
        return None

    # on teste peak dans l'espace [0..N) et aussi peak+N (utile si segment wrap)
    candidates = []
    for s, e in segs:
        if (s <= peak_idx < e) or (s <= peak_idx + N < e):
            candidates.append((s, e))

    if not candidates:
        return None

    # si plusieurs, on prend le plus "proche" / petit (ou le plus court)
    lengths = [e - s for s, e in candidates]
    return candidates[int(np.argmin(lengths))]
# ------------------------------------------------------------
# Utils: segments (bool -> listes d'intervalles), avec wrap
# ------------------------------------------------------------
def segments_from_bool(mask):
    """
    mask: array bool (N,)
    return: list of (start_idx, end_idx) inclusive-exclusive in [0,N],
            WITHOUT merging wrap. (wrap handled separately)
    """
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
    """
    Si un segment touche la fin et un autre touche le début, on merge en wrap.
    segs: list[(s,e)] avec 0<=s<e<=N
    return: list[(s,e)] mais e peut dépasser N si wrap-merge (ex: (350, 420))
    """
    if not segs:
        return []
    segs = sorted(segs, key=lambda x: x[0])

    # merge contigus/overlap (linéaire)
    merged = []
    cur_s, cur_e = segs[0]
    for s, e in segs[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))

    # wrap merge (dernier touche N et premier touche 0)
    if len(merged) >= 2 and merged[0][0] == 0 and merged[-1][1] == N:
        first_s, first_e = merged[0]
        last_s, last_e   = merged[-1]
        # on crée un segment "déroulé" : last_s -> (N + first_e)
        new_seg = (last_s, N + first_e)
        merged = merged[1:-1]
        merged.insert(0, new_seg)

    return merged


def pick_main_segment(segs):
    """Prend le segment le plus long (en 'déroulé' si wrap)."""
    if not segs:
        return None
    lengths = [e - s for s, e in segs]
    return segs[int(np.argmax(lengths))]


# ------------------------------------------------------------
# Normalisation + rephasage à l'extension max du coude (q[14])
# ------------------------------------------------------------
def normalize_emg_cycles_phased(emg, q_ref, peaks, n_points=200, min_len=10):
    """
    emg  : (n_muscles, n_frames_window)
    q_ref: (n_frames_window,) signal coude q[14] sur la même fenêtre
    peaks: indices (dans cette fenêtre) qui découpent les cycles
    return:
      emg_cycles_phased: (n_muscles, n_cycles, n_points) rephasé pour que q_ref max -> index 0
      q_cycles_phased  : (n_cycles, n_points) rephasé idem (utile debug)
      phase_idx        : (n_cycles,) index du max extension AVANT roll (sur le cycle normalisé)
    """
    emg = np.asarray(emg, float)
    q_ref = np.asarray(q_ref, float)

    n_muscles = emg.shape[0]
    emg_cycles = []
    q_cycles = []
    phase_idxs = []

    for i in range(len(peaks) - 1):
        i0, i1 = int(peaks[i]), int(peaks[i+1])
        if (i1 - i0) < min_len:
            continue

        seg_emg = emg[:, i0:i1]          # (m, L)
        seg_q   = q_ref[i0:i1]           # (L,)

        # resample sur n_points
        x_old = np.linspace(0, 1, seg_q.shape[0])
        x_new = np.linspace(0, 1, n_points)

        q_norm = np.interp(x_new, x_old, seg_q)  # (n_points,)
        emg_norm = np.zeros((n_muscles, n_points))
        for m in range(n_muscles):
            emg_norm[m] = np.interp(x_new, x_old, seg_emg[m])

        # index du max extension (q le + étendu)
        k0 = int(np.argmax(q_norm))
        phase_idxs.append(k0)

        # roll pour mettre ce max à 0° (index 0)
        q_phased = np.roll(q_norm, -k0)
        emg_phased = np.roll(emg_norm, -k0, axis=1)

        q_cycles.append(q_phased)
        emg_cycles.append(emg_phased)

    if len(emg_cycles) == 0:
        raise RuntimeError("Aucun cycle valide (phased).")

    emg_cycles_phased = np.stack(emg_cycles, axis=1)  # (m, cycle, time)
    q_cycles_phased   = np.stack(q_cycles, axis=0)    # (cycle, time)
    phase_idxs = np.asarray(phase_idxs, dtype=int)
    return emg_cycles_phased, q_cycles_phased, phase_idxs

def add_rotation_arrow(ax, clockwise=True, theta_deg=320, dtheta_deg=35, r=0.92, text=None):
    """
    Ajoute une flèche sur un axe polaire indiquant le sens de rotation.
    - theta_deg : angle de départ (degrés) où la flèche est placée
    - dtheta_deg: amplitude angulaire de la flèche
    - r         : rayon relatif (0..1) dans l'axe
    - clockwise : True -> sens horaire, False -> trigonométrique
    """
    # angles (en rad)
    th0 = np.deg2rad(theta_deg)
    dth = np.deg2rad(dtheta_deg) * (-1 if clockwise else 1)
    th1 = th0 + dth

    # Flèche en coordonnées (theta, r)
    ax.annotate(
        "",
        xy=(th1, r),
        xytext=(th0, r),
        arrowprops=dict(arrowstyle="-|>", lw=2),
        annotation_clip=False
    )

    if text is not None:
        ax.text(
            th0, r + 0.08,
            text,
            ha="center", va="center",
            fontsize=11
        )


def compute_mode_stats_phased(q, emg, q_index=14, distance=200, n_points=200):
    q_ref = q[q_index, :]
    peaks = detect_cycles_from_q(q_ref, distance=distance)

    emg_cyc, q_cyc, phase_idxs = normalize_emg_cycles_phased(
        emg, q_ref, peaks, n_points=n_points
    )

    mean = np.mean(emg_cyc, axis=1)  # (m, time)
    std  = np.std(emg_cyc, axis=1)

    return mean, std, emg_cyc, q_cyc, peaks, phase_idxs


# ------------------------------------------------------------
# Extraction onset/offset (par cycle) au-dessus d'un seuil
# ------------------------------------------------------------
def onset_offset_per_cycle(emg_cycles_phased, thr=0.30):
    """
    emg_cycles_phased: (m, n_cycles, N)
    return:
      onsets[m]  = array (n_cycles,) onset index
      offsets[m] = array (n_cycles,) offset index (peut dépasser N si wrap)
    """
    m, ncyc, N = emg_cycles_phased.shape
    onsets  = [ [] for _ in range(m) ]
    offsets = [ [] for _ in range(m) ]

    for mi in range(m):
        for ci in range(ncyc):
            y = emg_cycles_phased[mi, ci, :]
            peak_idx = int(np.argmax(y))

            mask = y > thr
            segs = segments_from_bool(mask)
            segs = merge_wrap_segments(segs, N)

            # ✅ choix stable: segment qui contient le pic EMG
            main = pick_segment_containing_peak(segs, peak_idx, N)

            if main is None:
                onsets[mi].append(np.nan)
                offsets[mi].append(np.nan)
                continue

            s, e = main
            onsets[mi].append(float(s))
            offsets[mi].append(float(e))

    return [np.array(v) for v in onsets], [np.array(v) for v in offsets]

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
mean_con, std_con, cycles_con, qcyc_con, peaks_con, phase_con = compute_mode_stats_phased(
    q_con, emg_con, q_index=14, distance=DISTANCE, n_points=N_POINTS
)

mean_ecc, std_ecc, cycles_ecc, qcyc_ecc, peaks_ecc, phase_ecc = compute_mode_stats_phased(
    q_ecc, emg_ecc, q_index=14, distance=DISTANCE, n_points=N_POINTS
)

print(f"Concentrique: {cycles_con.shape[1]} cycles")
print(f"Excentrique : {cycles_ecc.shape[1]} cycles")

# Normalise les cycles q_ref (ceux que tu as déjà récupérés: qref_con / qref_ecc)
q_cycles_con = normalize_q_cycles(q_con[14, :], peaks_con, n_points=N_POINTS)
q_cycles_ecc = normalize_q_cycles(q_ecc[14, :], peaks_ecc, n_points=N_POINTS)

x = np.linspace(0, 100, N_POINTS)

plt.figure(figsize=(12, 4))

# cycles individuels (alpha faible)
for c in q_cycles_con:
    plt.plot(x, c, alpha=0.15)
for c in q_cycles_ecc:
    plt.plot(x, c, alpha=0.15)

# moyennes (visibles)
plt.plot(x, q_cycles_con.mean(axis=0), linewidth=2.5, label=f"q_con[14] (moy, N={q_cycles_con.shape[0]})")
plt.plot(x, q_cycles_ecc.mean(axis=0), linewidth=2.5, label=f"q_ecc[14] (moy, N={q_cycles_ecc.shape[0]})")

plt.title("Vérification alignement des cycles (q[14] normalisé 0–100%)")
plt.xlabel("% cycle")
plt.ylabel("q[14] (a.u.)")
plt.grid(True, alpha=0.3)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()


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

# ============================================================
# PLOT "Fig.11 style": arcs d'activation au-dessus d'un seuil
# ============================================================
THR = 0.20  # 30%
N = N_POINTS
deg_per_idx = 360.0 / N

def plot_activation_arcs(ax, cycles_phased, muscle_names, thr=0.30, title=""):
    """
    ax: polar axis
    cycles_phased: (m, n_cycles, N)
    """
    m, ncyc, N = cycles_phased.shape
    deg_per_idx = 360.0 / N

    onsets, offsets = onset_offset_per_cycle(cycles_phased, thr=thr)

    # Style polaire
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)

    ring_h = 0.75
    r0s = np.arange(m)

    # Palette de couleurs (1 couleur / muscle)
    cmap = plt.get_cmap("tab20" if m <= 20 else "hsv")
    colors = [cmap(i % cmap.N) for i in range(m)]

    for mi in range(m):
        col = colors[mi]

        s = onsets[mi]
        e = offsets[mi]
        valid = np.isfinite(s) & np.isfinite(e)
        if np.sum(valid) < 2:
            continue

        s = s[valid]
        e = e[valid]

        mean_s = np.mean(s)
        mean_e = np.mean(e)
        std_s  = np.std(s)
        std_e  = np.std(e)

        def idx_to_rad(idx):
            return np.deg2rad(idx * deg_per_idx)

        # -------- arc épais (activation moyenne) --------
        th0 = idx_to_rad(mean_s)
        th1 = idx_to_rad(mean_e)
        t = np.linspace(th0, th1, 200)
        r = np.ones_like(t) * (r0s[mi] + ring_h)
        ax.plot(t, r, linewidth=7, color=col, solid_capstyle="round")

        # -------- arc fin (± écart-type) --------
        th0b = idx_to_rad(max(0.0, mean_s - std_s))
        th1b = idx_to_rad(min(N,   mean_e + std_e))
        tb = np.linspace(th0b, th1b, 200)
        rb = np.ones_like(tb) * (r0s[mi] + ring_h)
        ax.plot(tb, rb, linewidth=2.5, color=col, alpha=0.95)

    ax.set_title(title, pad=18)
    ax.set_yticks([])
    ax.set_ylim(-0.5, m + 0.8)
    ax.grid(True, alpha=0.25)

    # On retourne les couleurs pour la légende
    return colors

from matplotlib.lines import Line2D

fig = plt.figure(figsize=(14, 7))

ax1 = fig.add_subplot(1, 2, 1, projection="polar")
ax2 = fig.add_subplot(1, 2, 2, projection="polar")

colors = plot_activation_arcs(
    ax1, cycles_con, muscle_names,
    thr=THR,
    title=f"Concentrique – seuil {int(THR*100)} %"
)

plot_activation_arcs(
    ax2, cycles_ecc, muscle_names,
    thr=THR,
    title=f"Excentrique – seuil {int(THR*100)} %"
)

# Flèches sens de rotation
add_rotation_arrow(ax1, clockwise=False,  theta_deg=340, dtheta_deg=40, r=15)
add_rotation_arrow(ax2, clockwise=True, theta_deg=20, dtheta_deg=40, r=15)

# -------------------------
# LÉGENDE GLOBALE EN DESSOUS
# -------------------------
legend_handles = [
    Line2D([0], [0], color=colors[i], lw=5, label=muscle_names[i])
    for i in range(len(muscle_names))
]

fig.legend(
    handles=legend_handles,
    loc="lower center",
    ncol=4,                      # ajuste selon nb muscles
    frameon=True,
    fontsize=11,
    bbox_to_anchor=(0.5, -0.02)  # bien en dessous
)

fig.suptitle(
    f"Activations EMG > {int(THR*100)} % – repère 0° = extension max du coude (q[14])",
    fontsize=15,
    y=0.97
)

plt.tight_layout(rect=[0, 0.08, 1, 0.93])
plt.show()