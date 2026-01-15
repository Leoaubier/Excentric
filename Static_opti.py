from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

import biorbd
print(dir(biorbd))  # tu devrais voir BiorbdModel ou Biorbd



#
# This examples shows how to
#     1. Load a model with muscles
#     2. Position the model at a chosen position (Q) and velocity (Qdot)
#     3. Define a target generalized forces (Tau)
#     4. Compute the muscle activations that reproduce this Tau (Static optimization)
#     5. Print them to the console
#
# Please note that this example will work only with the Eigen backend
#

def main():
    # Load a predefined model
    MODEL_PATH = "/Users/leo/Desktop/Projet/modele_opensim/wu_bras_gauche_Sidonie_last.bioMod"
    Q_PATH = "/Users/leo/Desktop/Projet/Collecte_25_11/IK/q_inverse_kinematic_sidonie_40W.npy"
    QDOT_PATH = "/Users/leo/Desktop/Projet/Collecte_25_11/IK/qdot_inverse_kinematic_sidonie_40W.npy"
    TAU_PATH = "/Users/leo/Desktop/Projet/Collecte_25_11/ID/tau_inverse_dynamic_Sidonie_40w.npy"

    model = biorbd.Biorbd(MODEL_PATH)

    Start = 3000
    End = 3005
    n_frames = End - Start

    q_n = np.load(Q_PATH)
    qdot_n = np.load(QDOT_PATH)
    #qdot_n = np.gradient(q_n,1/100, axis=1)


    tau_n = np.load(TAU_PATH)

    #tau_n[0:6,:] = 0 # mise à 0 des 6 premiers DoF
    #tau_n[14:,:] = 0 # mise à 0 des 4 derniers DoF


    q = [0 for _ in range(n_frames)]
    qdot = [0 for _ in range(n_frames)]
    tau = [0 for _ in range(n_frames)]

    for i in range(n_frames):
        q[i] = q_n[:,i+Start]
        qdot[i] = qdot_n[:,i+Start]
        tau[i] = tau_n[:,i+Start]


    optim = biorbd.StaticOptimization(model)
    muscle_activations = []
    tau_residual = []

    for value in optim.perform_frames(q, qdot, tau):
        muscle_activations.append(value)



    # Print them to the console
    for i, activations in enumerate(muscle_activations):
        print(f"Frame {i}: {activations}")

    A = np.array(muscle_activations).T  # (n_muscles, n_frames)

    np.save("/Users/leo/Desktop/Projet/Collecte_25_11/statique/muscle_activations_biorbd.npy",A)


    fig, axes = plt.subplots(7, 5, figsize=(4 * 7, 3 * 5), sharex=True)
    axes = axes.flatten()  # pour parcourir facilement

    for ax, (i, muscle_name) in zip(axes, enumerate(model.muscles)):
        ax.plot(A[i])
        ax.set_title(muscle_name, fontsize=8)
        ax.set_ylabel("Activation", fontsize=6)
        ax.grid(True)

    # Désactiver les subplots inutilisés si nb muscles < 35
    for ax in axes[len(A):]:
        ax.axis("off")

    axes[-1].set_xlabel("Frame")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
