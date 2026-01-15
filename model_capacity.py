import numpy as np
import biorbd
import matplotlib.pyplot as plt


# ----------------------------
# Config
# ----------------------------
MODEL_PATH = "/Users/leo/Desktop/Projet/modele_opensim/wu_bras_gauche_seth_left_Sidonie.bioMod"
ACTIVE_DOF = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]  # DoF à considérer

def compute_muscle_forces_from_activation(model: biorbd.Model, q: np.ndarray, qdot: np.ndarray, a: np.ndarray):
    a = np.asarray(a, dtype=float).reshape(-1)
    nb_mus = model.nbMuscles()
    if a.shape[0] != nb_mus:
        raise ValueError(f"Activation size {a.shape[0]} != nbMuscles {nb_mus}")

    states = model.stateSet()
    for i in range(nb_mus):
        states[i].setActivation(float(a[i]))

    forces = model.muscleForces(states, q, qdot).to_array()

    return np.asarray(forces).reshape(-1)
# ----------------------------
# Fonction diagnostic couple max
# ----------------------------
def dynamic_max_muscle_couples(model, q_list, qdot_list, active_dof=ACTIVE_DOF):
    nbMus = model.nbMuscles()
    n_frames = q_list.shape[1]
    tau_max_all = np.zeros((len(active_dof), n_frames))
    tau_min_all = np.zeros((len(active_dof), n_frames))

    for k in range(n_frames):
        qk = q_list[:, k]
        qdotk = qdot_list[:, k]

        # Forces musculaires dynamiques pour activations unitaires individuelles
        R = model.musclesLengthJacobian(qk).to_array().T  # (nbQ, nbMus)

        for i, dof_idx in enumerate(active_dof):
            # Identifier muscles agonistes et antagonistes pour ce DoF
            muscles_pos = np.where(R[dof_idx, :] > 0)[0]
            muscles_neg = np.where(R[dof_idx, :] < 0)[0]

            # Max couple positif : n'activer que les agonistes
            a_pos = np.zeros(nbMus)
            a_pos[muscles_pos] = 1.0
            f_pos = compute_muscle_forces_from_activation(model, qk, qdotk, a_pos)
            tau_max_all[i, k] = np.sum(R[dof_idx, :] * f_pos)
            #tau_max_all[i, k] = model.muscularJointTorque(qk, qdotk, a_pos).to_array()[dof_idx]

            # Max couple négatif : n'activer que les antagonistes
            a_neg = np.zeros(nbMus)
            a_neg[muscles_neg] = 1.0
            f_neg = compute_muscle_forces_from_activation(model, qk, qdotk, a_neg)
            tau_min_all[i, k] = np.sum(R[dof_idx, :] * f_neg)
            #tau_min_all[i, k] = model.muscularJointTorque(qk, qdotk, a_neg).to_array()[dof_idx]

    return {
        "tau_max": tau_max_all,
        "tau_min": tau_min_all,
        "active_dof": active_dof,
        "n_frames": n_frames
    }


# --------------------------------------------------------
# Comparaison avec tau consigne
# --------------------------------------------------------
def compare_tau_dynamic(tau_max_dict, tau_target, active_dof=ACTIVE_DOF):
    tau_max = tau_max_dict["tau_max"]
    tau_min = tau_max_dict["tau_min"]
    n_frames = tau_max_dict["n_frames"]

    for k in range(n_frames):
        print(f"\n=== Frame {k} ===")
        for i, dof_idx in enumerate(active_dof):
            target = tau_target[i, k]
            tmax = tau_max[i, k]
            tmin = tau_min[i, k]
            achievable = (target <= tmax) and (target >= tmin)
            status = "OK" if achievable else "NON REALISABLE"
            print(f"DoF {dof_idx:2d} | tau_target={target:6.2f} | "
                  f"tau_min={tmin:6.2f}, tau_max={tmax:6.2f} -> {status}")


def plot_tau_feasibility(tau_dyn_dict, tau_target,q, active_dof=ACTIVE_DOF):
    """
    Visualise la faisabilité du couple musculaire dynamique.

    Args:
        tau_dyn_dict (dict): sortie de dynamic_max_muscle_couples
        tau_target (np.ndarray): couples consigne, shape (nbDoF, n_frames)
        active_dof (list): indices des DoF
    """
    tau_max = tau_dyn_dict["tau_max"]
    tau_min = tau_dyn_dict["tau_min"]
    n_frames = tau_dyn_dict["n_frames"]

    x = np.arange(n_frames)

    n_dof = len(active_dof)
    n_cols = 3
    n_rows = int(np.ceil(n_dof / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 3 * n_rows), sharex=True)
    axes = axes.flatten()

    for i, dof_idx in enumerate(active_dof):
        ax = axes[i]
        # Bande réalisable
        ax.fill_between(x, tau_min[i, :], tau_max[i, :], color='lightblue', alpha=0.4, label='Tau réalisable')

        if i == 14:
            ax.plot(x, q[i,:]*10, label="flexion coude")

        # Tau cible
        tau_cible = tau_target[i, :]
        # couleur selon réalisabilité
        color_line = np.where((tau_cible >= tau_min[i, :]) & (tau_cible <= tau_max[i, :]), 'green', 'red')
        for j in range(n_frames):
            ax.plot(j, tau_cible[j], 'o', color=color_line[j])

        ax.set_title(f"DoF {dof_idx}")
        ax.set_ylabel("Couple (Nm)")
        ax.grid(True)

    # Supprimer axes vides
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.xlabel("Frame")
    plt.suptitle("Faisabilité des couples musculaires dynamiques\nVert = OK, Rouge = non réalisable")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


model = biorbd.Model(MODEL_PATH)
Q_PATH = "/Users/leo/Desktop/Projet/Collecte_25_11/IK/q_inverse_kinematic_sidonie_40W.npy"
QDOT_PATH = "/Users/leo/Desktop/Projet/Collecte_25_11/IK/qdot_inverse_kinematic_sidonie_40W.npy"
TAU_PATH = "/Users/leo/Desktop/Projet/Collecte_25_11/ID/tau_inverse_dynamic_Sidonie_40w.npy"


# Frames à tester
frames_to_check = [1000, 1050, 1500, 1520, 2000]
q_data = np.load(Q_PATH)[:, 3000:4000]
qdot_data = np.load(QDOT_PATH)[:, 3000:4000]
tau_target = np.load(TAU_PATH)[ACTIVE_DOF, :][:, 3000:4000]

# Calcul dynamique
tau_dyn_dict = dynamic_max_muscle_couples(model, q_data, qdot_data, ACTIVE_DOF)

# Comparaison avec couple consigne
compare_tau_dynamic(tau_dyn_dict, tau_target, ACTIVE_DOF)

plot_tau_feasibility(tau_dyn_dict, tau_target, q_data, ACTIVE_DOF)
