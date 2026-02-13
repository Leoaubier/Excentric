import numpy as np
import biorbd
import matplotlib.pyplot as plt
from xarray.ufuncs import rad2deg
from math import pi


MODEL_PATH = ("/Users/leo/Desktop/Projet/modele_opensim/wu_bras_gauche_seth_left_Sidonie_vtp.bioMod")

DOF_ELBOW = 14          # DoF coude
MUSC_BIC = 33           # biceps (index)

ELBOW_FLEX_RAD = np.deg2rad(30)   # posture fixe : coude fléchi ~70°
BIC_ACTIVATION = 1.0              # activation biceps (le reste = 0)

QD_ABS_MAX = 25.0                  # rad/s
N_PER_SIDE = 60                   # nb points <0 et >0 (0 inclus en plus)

# grille symétrique autour de 0 : ecc<0, iso=0, con>0
qd_neg = np.linspace(-QD_ABS_MAX, -QD_ABS_MAX / N_PER_SIDE, N_PER_SIDE)  # exclut 0
qd_pos = np.linspace(QD_ABS_MAX / N_PER_SIDE, QD_ABS_MAX, N_PER_SIDE)    # exclut 0
qd_grid = np.concatenate([qd_neg, [0.0], qd_pos])                        # inclut 0


def compute_muscle_forces_from_activation(model: biorbd.Model,
                                         q: np.ndarray,
                                         qdot: np.ndarray,
                                         a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float).reshape(-1)
    nb_mus = model.nbMuscles()
    if a.shape[0] != nb_mus:
        raise ValueError(f"Activation size {a.shape[0]} != nbMuscles {nb_mus}")

    states = model.stateSet()
    model.updateMuscles(q, qdot)

    for i in range(nb_mus):
        states[i].setActivation(float(a[i]))

    forces = model.muscleForces(states, q, qdot).to_array()
    return np.asarray(forces).reshape(-1)

#main
model = biorbd.Model(MODEL_PATH)

# posture fixe (tous q = 0 sauf coude fléchi)
q = np.zeros(model.nbQ(), dtype=float)
if not (0 <= DOF_ELBOW < q.shape[0]):
    raise ValueError(f"DOF_ELBOW={DOF_ELBOW} out of range [0, {q.shape[0]-1}]")
q[DOF_ELBOW] = ELBOW_FLEX_RAD

# activation: biceps seul
a = np.zeros(model.nbMuscles(), dtype=float)
if not (0 <= MUSC_BIC < a.shape[0]):
    raise ValueError(f"MUSC_BIC={MUSC_BIC} out of range [0, {a.shape[0]-1}]")
a[MUSC_BIC] = BIC_ACTIVATION

# qdot: partout 0 sauf au coude
qdot0 = np.zeros(model.nbQdot(), dtype=float)
if DOF_ELBOW >= qdot0.shape[0]:
    raise ValueError(
        f"DOF_ELBOW={DOF_ELBOW} dépasse nbQdot={qdot0.shape[0]}. "
        "Vérifie l'indexation q/qdot."
    )

force_bic = np.zeros_like(qd_grid, dtype=float)

for i, qd in enumerate(qd_grid):
    qdot = qdot0.copy()
    qdot[DOF_ELBOW] = qd

    f = compute_muscle_forces_from_activation(model, q, qdot, a)
    force_bic[i] = f[MUSC_BIC]


# PLOT

fig, ax = plt.subplots(1, 1, figsize=(8, 4))
ax.plot(qd_grid, force_bic, linewidth=2)
ax.axvline(0.0, linestyle="--", linewidth=1)

# zones : ecc / iso / con
ymin, ymax = ax.get_ylim()
ax.axvspan(qd_grid.min(), 0.0, alpha=0.10)   # excentrique
ax.axvspan(0.0, qd_grid.max(), alpha=0.10)   # concentrique
ax.text(0.02, 0.95, "Excentrique (qdot < 0)", transform=ax.transAxes, va="top")
ax.text(0.50, 0.95, "Isométrique (qdot = 0)", transform=ax.transAxes, va="top", ha="center")
ax.text(0.98, 0.95, "Concentrique (qdot > 0)", transform=ax.transAxes, va="top", ha="right")

ax.set_title(f"Force biceps ({model.muscleNames()[MUSC_BIC].to_string()}) en fonction de qdot coude (DoF {DOF_ELBOW}), flexion de {rad2deg(ELBOW_FLEX_RAD)}°")
ax.set_xlabel("qdot coude (rad.s-1)")
ax.set_ylabel("Force musculaire (N)")
ax.grid(True)
plt.tight_layout()
plt.show()

print(f"Force isométrique (qdot=0): {force_bic[np.where(qd_grid==0.0)[0][0]]:.3f} N")
