import numpy as np
from scipy.optimize import minimize
import biorbd
import matplotlib.pyplot as plt
import numpy as np


def plot_tau_envelope(results, dof_list, frame_range=None):
    """
    results : dict retourné par compare_capacity_to_ID
    dof_list : liste des DoF à tracer
    """

    n_dof = len(dof_list)
    fig, axes = plt.subplots(n_dof, 1, figsize=(12, 3 * n_dof), sharex=True)

    if n_dof == 1:
        axes = [axes]

    for ax, dof in zip(axes, dof_list):
        tau_max = results[dof]["tau_max"]
        tau_min = results[dof]["tau_min"]
        tau_ID  = results[dof]["tau_ID"]

        if frame_range is not None:
            tau_max = tau_max[frame_range]
            tau_min = tau_min[frame_range]
            tau_ID  = tau_ID[frame_range]
            frames  = np.arange(len(frame_range))
        else:
            frames = np.arange(len(tau_ID))

        # Enveloppe musculaire
        ax.fill_between(
            frames,
            tau_min,
            tau_max,
            color="lightblue",
            alpha=0.5,
            label="Capacité musculaire"
        )

        # Faisabilité
        feasible = (tau_ID >= tau_min) & (tau_ID <= tau_max)

        ax.plot(frames[feasible], tau_ID[feasible],
                color="green", linewidth=2, label="τ ID réalisable")

        ax.scatter(frames[~feasible], tau_ID[~feasible],
                   color="red", s=20, label="τ ID non réalisable")

        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel(f"DoF {dof} (Nm)")
        ax.grid(True)

    axes[0].legend(loc="upper right")
    axes[-1].set_xlabel("Frame")

    plt.tight_layout()
    plt.show()


import numpy as np
import biorbd
from scipy.optimize import minimize

def maximize_tau_dof_biorbd(model, q, qdot, dof_idx, sign=1, a0=None):
    """
    Optimise les activations musculaires pour maximiser le couple d'un DoF donné
    en utilisant le vrai muscularJointTorque de Biorbd.

    Parameters
    ----------
    model : biorbd.Model
        Modèle musculo-squelettique
    q : array (nbQ,)
        Positions articulaires
    qdot : array (nbQ,)
        Vitesses articulaires
    dof_idx : int
        Index du DoF à maximiser
    sign : int
        +1 → max positif, -1 → max négatif
    a0 : array
        Point de départ des activations (optionnel)
    """
    nb_mus = model.nbMuscles()
    if a0 is None:
        a0 = np.ones(nb_mus) * 0.5  # point de départ moyen

    # borne activations [0,1]
    bounds = [(0,1)]*nb_mus

    def objective(a):
        tau = model.muscularJointTorque(q, qdot, a).to_array()
        # on maximise tau[dof_idx] dans le sens sign
        return -sign * tau[dof_idx]

    res = minimize(objective, a0, method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 200, 'ftol':1e-6})

    tau_opt = model.muscularJointTorque(q, qdot, res.x).to_array()
    return {
        "tau_max": tau_opt[dof_idx],
        "activations": res.x,
        "success": res.success
    }



def compare_capacity_to_ID_biorbd(model, q_data, qdot_data, tau_ID, active_dof):
    n_frames = q_data.shape[1]
    results = {}

    for dof in active_dof:
        tau_max = np.zeros(n_frames)
        tau_min = np.zeros(n_frames)

        a0 = np.ones(model.nbMuscles())*0.5  # warm-start
        for k in range(n_frames):
            qk = q_data[:, k]
            qdotk = qdot_data[:, k]

            pos = maximize_tau_dof_biorbd(model, qk, qdotk, dof, sign=+1, a0=a0)
            neg = maximize_tau_dof_biorbd(model, qk, qdotk, dof, sign=-1, a0=a0)

            tau_max[k] = pos["tau_max"]
            tau_min[k] = neg["tau_max"]

            # réutiliser les activations comme point de départ pour la frame suivante
            a0 = pos["activations"]

        results[dof] = {
            "tau_max": tau_max,
            "tau_min": tau_min,
            "tau_ID": tau_ID[dof, :]
        }

    return results



ACTIVE_DOF = [6, 7, 8, 9, 10, 11, 12, 13, 14]
MODEL_PATH = "/Users/leo/Desktop/Projet/modele_opensim/wu_bras_gauche_seth_left_Sidonie"

Q_PATH = "/Users/leo/Desktop/Projet/Collecte_25_11/IK/q_inverse_kinematic_sidonie_40W.npy"
QDOT_PATH = "/Users/leo/Desktop/Projet/Collecte_25_11/IK/qdot_inverse_kinematic_sidonie_40W.npy"
TAU_PATH = "/Users/leo/Desktop/Projet/Collecte_25_11/ID/tau_inverse_dynamic_Sidonie_40w.npy"



model = biorbd.Model(MODEL_PATH)

n_musc = model.nbMuscles()
n_dof = model.nbDof()
q = np.load(Q_PATH)
qdot = np.load(QDOT_PATH)
tau = np.load(TAU_PATH)

results = compare_capacity_to_ID_biorbd(
    model,
    q[:,3000:3200],
    qdot[:,3000:3200],
    tau[:,3000:3200],
    ACTIVE_DOF
)

plot_tau_envelope(results, ACTIVE_DOF)
