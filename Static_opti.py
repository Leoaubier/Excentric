from pathlib import Path
import numpy as np
import biorbd

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

    model = biorbd.Biorbd(f"/Users/leo/Desktop/Projet/modele_opensim/wu_bras_gauche_seth_left_Sidonie.bioMod")
    q = np.load("/Users/leo/Desktop/Projet/Collecte_25_11/IK/q_inverse_kinematic_sidonie_40W.npy")
    qdot = np.load("/Users/leo/Desktop/Projet/Collecte_25_11/IK/qdot_inverse_kinematic_sidonie_40W.npy")
    qddot = np.load("/Users/leo/Desktop/Projet/Collecte_25_11/IK/qddot_inverse_kinematic_sidonie_40W.npy")
    tau = np.load("/Users/leo/Desktop/Projet/Collecte_25_11/ID/tau_inverse_dynamic_Sidonie_40w.npy")

    # Choose a position/velocity/torque to compute muscle activations from.
    # If only one frame the Vector are not mandatory and the Static Optimization function can be called
    # directly with numpy arrays
    q_list = [q[:, i] for i in range(q.shape[1])]
    qdot_list = [qdot[:, i] for i in range(qdot.shape[1])]
    tau_list = [tau[:, i] for i in range(tau.shape[1])]


    # Proceed with the static optimization. When perform is called, all the frames are processed at once, even though
    # it is a loop. That is so the initial guess is dependent of the previous frame. So the first "frame" of the loop is
    # very long (as it computes everythin). Then, the following frames are very fast (as it only returns the precomputed
    # results)
    optim = biorbd.StaticOptimization(model)
    muscle_activations = []
    for value in optim.perform_frames(q_list, qdot_list, tau_list):
        muscle_activations.append(value)

    # Print them to the console
    for i, activations in enumerate(muscle_activations):
        print(f"Frame {i}: {activations}")


if __name__ == "__main__":
    main()