import numpy as np
import biorbd
import matplotlib.pyplot as plt

MODE_PEDALAGE = "eccentric"
PUISSANCE = "40"

def compute_muscle_forces_from_activation(model: biorbd.Model, q: np.ndarray, qdot: np.ndarray, a: np.ndarray):
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

def get_R(model, q):
    nb_mus = model.nbMuscles()

    # Jacobien des longueurs musculaires
    J = model.musclesLengthJacobian(q).to_array()   # (nbMus, nbQ)
    R = -J.T                                        # (nbQ, nbMus) --> -J normalement

    return R

A_PATH = f"/Users/leo/Desktop/Projet/Collecte_25_11/{MODE_PEDALAGE}_{PUISSANCE}W/muscle_activations_nonlinear.npy"
MODEL_PATH = "/Users/leo/Desktop/Projet/modele_opensim/wu_bras_gauche_seth_left_Sidonie.bioMod"

Q_PATH = f"/Users/leo/Desktop/Projet/Collecte_25_11/{MODE_PEDALAGE}_{PUISSANCE}W/q_inverse_kinematic.npy"
QDOT_PATH = f"/Users/leo/Desktop/Projet/Collecte_25_11/{MODE_PEDALAGE}_{PUISSANCE}W/qdot_inverse_kinematic.npy"
TAU_PATH = f"/Users/leo/Desktop/Projet/Collecte_25_11/{MODE_PEDALAGE}_{PUISSANCE}W/tau_inverse_dynamic.npy"

FIRST, END = 3000, 4000


model = biorbd.Model(MODEL_PATH)

n_musc = model.nbMuscles()
n_dof = model.nbDof()
a = np.load(A_PATH)
q = np.load(Q_PATH)
qdot = np.load(QDOT_PATH)
tau = np.load(TAU_PATH)

n_frames = END - FIRST

f_full = np.zeros((n_musc,n_frames))


tau_act = np.zeros((n_dof,n_frames))
tau_verif = np.zeros((n_dof,n_frames))

for k in range(n_frames):
    qk = q[:, k+FIRST].reshape(-1)
    qdotk = qdot[:, k+FIRST].reshape(-1)
    tauk = tau[:, k+FIRST].reshape(-1)
    ak = a[:, k].reshape(-1)

    f = compute_muscle_forces_from_activation(model, qk, qdotk, ak)

    R = get_R(model, qk)

    tau_act[:,k] = R @ f

    f_full[:,k] = f

    tau_from_contrib = np.sum(R * f[None, :], axis=1)
    print(np.allclose(tau_from_contrib, tau_act[:, k]))

    tau_verif[:,k] = model.muscularJointTorque(qk, qdotk, ak).to_array()


for i in range(n_dof):
    plt.plot(tau_act[i,:], label = "tau activations")
    plt.plot(tau_verif[i,:], label = "tau activation via model")
    plt.plot(tau[i,FIRST:END], label = "tau")
    plt.legend()
    plt.show()

fig, axes = plt.subplots(7, 5, figsize=(4 * 7, 3 * 5), sharex=True)
axes = axes.flatten()  # pour parcourir facilement

for ax, (i, muscle_name) in zip(axes, enumerate(model.muscleNames())):
    ax.plot(f_full[i,:])
    ax.set_title(muscle_name.to_string(), fontsize=8)
    ax.set_ylabel("force musculaire", fontsize=6)
    ax.grid(True)

# Désactiver les subplots inutilisés si nb muscles < 35
for ax in axes[n_musc:]:
    ax.axis("off")

axes[-1].set_xlabel("Frame")
plt.tight_layout()
plt.show()