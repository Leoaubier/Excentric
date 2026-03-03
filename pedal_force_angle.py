import numpy as np
import matplotlib.pyplot as plt

# =========================
# SETTINGS
# =========================
PUISSANCE = "80"

BASE_CON = f"/Users/leo/Desktop/Projet/Collecte_25_11/concentric_{PUISSANCE}W"
BASE_ECC = f"/Users/leo/Desktop/Projet/Collecte_25_11/eccentric_{PUISSANCE}W"

CONSTRAINT_CON_PATH = f"{BASE_CON}/constraint_crank.npy"   # [Moment(3,T), Force(3,T)] repère crank
CRANK_CON_PATH      = f"{BASE_CON}/crank_angle.npy"        # (T,)

CONSTRAINT_ECC_PATH = f"{BASE_ECC}/constraint_crank.npy"
CRANK_ECC_PATH      = f"{BASE_ECC}/crank_angle.npy"

N_POINTS = 360
MIN_CYCLE_FRAMES = 30

if PUISSANCE == "40":
    START_CON = 2000  # frame de début (ex : 2000)
    END_CON = 6000  # frame de fin
    START_ECC = 2000  # frame de début (ex : 2000)
    END_ECC = 5000  # frame de fin
elif PUISSANCE == "60":
    START_CON = 2000  # frame de début (ex : 2000)
    END_CON = 5000  # frame de fin
    START_ECC = 1500  # frame de début (ex : 2000)
    END_ECC = 3500  # frame de fin
elif PUISSANCE == "80":
    START_CON = 1500  # frame de début (ex : 2000)
    END_CON = 4000  # frame de fin
    START_ECC = 7000  # frame de début (ex : 2000)
    END_ECC = 10000  # frame de fin
else:
    print("PB PUISSANCE")


def ensure_forward_rotation(crank_angle, *signals):
    crank_angle = np.asarray(crank_angle, float)

    if np.median(np.diff(crank_angle)) < 0:
        crank_angle = crank_angle[::-1]
        signals = [s[..., ::-1] for s in signals]
        print("Rotation inversée détectée → ECC remis dans le sens croissant")

    return (crank_angle, *signals)
# ============================================================
# Forcer même origine angulaire
# ============================================================
def set_common_angle_origin(crank_angle):
    """
    Force le premier échantillon à 0 rad
    et remet tout dans [0, 2π]
    """
    crank_angle = np.unwrap(np.asarray(crank_angle, float))
    crank_angle = crank_angle - crank_angle[0]
    crank_angle = np.mod(crank_angle, 2*np.pi)
    return crank_angle
# =========================
# Cycle detection from crank angle (wrap 2pi)
# =========================
def detect_cycles_from_crank(crank_angle, min_cycle_frames=30):
    a = np.asarray(crank_angle, float)
    da = np.diff(a)
    threshold = np.pi

    if np.median(da) > 0:
        wraps = np.where(da < -threshold)[0] + 1
    else:
        wraps = np.where(da > threshold)[0] + 1

    if wraps.size == 0:
        raise RuntimeError("Aucun wrap détecté dans crank_angle.")

    good = [wraps[0]]
    for s in wraps[1:]:
        if s - good[-1] >= min_cycle_frames:
            good.append(s)

    starts = np.array(good, dtype=int)
    if starts.size < 2:
        raise RuntimeError("Pas assez de cycles détectés.")
    return starts


# =========================
# Normalisation par angle
# =========================
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
        seg_a = a[i0:i1] - a[i0]

        seg_phi = np.mod(seg_a, 2*np.pi)
        order = np.argsort(seg_phi)
        seg_phi = seg_phi[order]
        seg_s   = seg_s[:, order]

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


# =========================
# LOAD helper (+ cropping)
# =========================
def load_force_and_angle(constraint_path, crank_path, first, end):
    constraint = np.load(constraint_path, allow_pickle=True)
    if isinstance(constraint, np.ndarray) and constraint.dtype == object:
        _, F = constraint[0], constraint[1]
    else:
        _, F = constraint[0], constraint[1]

    F = np.asarray(F, float)  # (3,T)
    crank = np.asarray(np.load(crank_path), float).reshape(-1)

    T = min(F.shape[1], crank.shape[0])
    F = F[:, :T]
    crank = crank[:T]

    # crop
    first = max(0, int(first))
    end = min(int(end), T) if end is not None else T
    if end <= first:
        raise ValueError(f"Bad crop range: first={first}, end={end}, T={T}")

    return F[:, first:end], crank[first:end], first, end


def compute_cycle_stats(F_pedal, crank_angle):
    Fx, Fy, Fz = F_pedal[0, :], F_pedal[1, :], F_pedal[2, :]


    signals = np.vstack([Fx, Fy, Fz])  # (3,T)

    starts = detect_cycles_from_crank(crank_angle, min_cycle_frames=MIN_CYCLE_FRAMES)
    cycles, angle_grid = normalize_cycles_by_crank(signals, crank_angle, starts, n_points=N_POINTS)

    mean = cycles.mean(axis=1)  # (3,n_points)
    std  = cycles.std(axis=1)
    return mean, std, angle_grid, cycles.shape[1], starts


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


# =========================
# MAIN
# =========================
F_con, crank_con, f0c, f1c = load_force_and_angle(CONSTRAINT_CON_PATH, CRANK_CON_PATH, START_CON, END_CON)
F_ecc, crank_ecc, f0e, f1e = load_force_and_angle(CONSTRAINT_ECC_PATH, CRANK_ECC_PATH, START_ECC, END_ECC)

crank_ecc, F_ecc = ensure_forward_rotation(
       crank_ecc, F_ecc
    )

#crank_con = set_common_angle_origin(crank_con)
#crank_ecc = set_common_angle_origin(crank_ecc)

mean_con, std_con, angle_grid, ncyc_con, starts_con = compute_cycle_stats(F_con, crank_con)
mean_ecc, std_ecc, angle, ncyc_ecc, starts_ecc = compute_cycle_stats(F_ecc, crank_ecc)

x_deg = (np.rad2deg(angle_grid) % 360)

print(f"Concentrique: frames [{f0c}:{f1c}] -> N cycles utilisés = {ncyc_con}")
print(f"Excentrique : frames [{f0e}:{f1e}] -> N cycles utilisés = {ncyc_ecc}")

# Plot crank + starts (diagnostic)
plot_crank_with_starts(crank_con, starts_con, title=f"Concentrique {PUISSANCE}W — crank_angle (crop {f0c}:{f1c}) + starts")
plot_crank_with_starts(crank_ecc, starts_ecc, title=f"Excentrique {PUISSANCE}W — crank_angle (crop {f0e}:{f1e}) + starts")

# Plot comparison (2 subplots)
fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)

ax = axes[0]
ax.plot(x_deg, mean_con[0], linewidth=2)
ax.fill_between(x_deg, mean_con[0]-std_con[0], mean_con[0]+std_con[0], alpha=0.20)
ax.plot(x_deg, mean_ecc[0], linewidth=2)
ax.fill_between(x_deg, mean_ecc[0]-std_ecc[0], mean_ecc[0]+std_ecc[0], alpha=0.20)
ax.set_title("Force repère pédalier : Effort normal : Fx (moyenne ± SD par cycle)")
ax.set_ylabel("Fx (N)")
ax.set_xlim(0, 360)
ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(x_deg, mean_con[1], linewidth=2)
ax.fill_between(x_deg, mean_con[1]-std_con[1], mean_con[1]+std_con[1], alpha=0.20)
ax.plot(x_deg, mean_ecc[1], linewidth=2)
ax.fill_between(x_deg, mean_ecc[1]-std_ecc[1], mean_ecc[1]+std_ecc[1], alpha=0.20)
ax.set_title("Force repère pédalier : Effort axial : Fy (moyenne ± SD par cycle)")
ax.set_ylabel("Fy (N)")
ax.set_xlim(0, 360)
ax.grid(True, alpha=0.3)

ax = axes[2]
ax.plot(x_deg, mean_con[2], linewidth=2, label=f"Concentrique (N={ncyc_con})")
ax.fill_between(x_deg, mean_con[2]-std_con[2], mean_con[2]+std_con[2], alpha=0.20)
ax.plot(x_deg, mean_ecc[2], linewidth=2, label=f"Excentrique (N={ncyc_ecc})")
ax.fill_between(x_deg, mean_ecc[2]-std_ecc[2], mean_ecc[2]+std_ecc[2], alpha=0.20)
ax.set_title("Force repère pédalier : Effort tangentiel : Fz (moyenne ± SD par cycle)")
ax.set_xlabel("Angle pédalier (deg)")
ax.set_ylabel("Fz (N)")
ax.set_xlim(0, 360)
ax.grid(True, alpha=0.3)
ax.legend(loc="lower right")

plt.tight_layout()
plt.show()
