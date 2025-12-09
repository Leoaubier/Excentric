from pathlib import Path
import numpy as np
import biorbd
import matplotlib.pyplot as plt

#
# This examples shows how to
#     1. Load a model
#     2. Position the model at a chosen position (Q), velocity (Qdot) and acceleration (Qddot)
#     3. Compute the generalized forces (tau) at this state (inverse dynamics)
#     4. Print them to the console
#
# Please note that this example will work only with the Eigen backend
#
model_path = "/Users/leo/Desktop/Projet/modele_opensim/wu_bras_gauche_seth_left_Sidonie.bioMod"
model_pedal_path = '/Users/leo/Desktop/Projet/modele_opensim/model_pedal.bioMod'
q_path     = "/Users/leo/Desktop/Projet/Collecte_25_11/IK/q_inverse_kinematic_sidonie_40W.npy"
q_pedal_path = "/Users/leo/Desktop/Projet/Collecte_25_11/IK/inverse_kinematic_pedal_40W.npy"
qdot_path  = "/Users/leo/Desktop/Projet/Collecte_25_11/IK/qdot_inverse_kinematic_sidonie_40W.npy"
qddot_path  = "/Users/leo/Desktop/Projet/Collecte_25_11/IK/qddot_inverse_kinematic_sidonie_40W.npy"
force_path = "/Users/leo/Desktop/Projet/Collecte_25_11/IK/constraint_global_40W.npy"

def inverse_dynamic(model_path, q_path, qdot_path, qddot_path):
    current_file_dir = Path(__file__).parent
    model = biorbd.Biorbd(model_path)
    force = np.load(force_path)

    nq = model.nb_q
    print("DoF du modèle :", nq)


    q_recons = np.load(q_path)
    qdot_recons = np.load(qdot_path)
    qddot_recons = np.load(qddot_path)
    tau = np.zeros((nq, int(q_recons.shape[1])))

    origin = np.zeros((3, q_recons.shape[1]))
    #print(origin.shape)
    q_pedal = np.load(q_pedal_path)
    mod_ped = biorbd.Biorbd(model_pedal_path)

    point_app = np.zeros(3) #point d'application dans le repère pédale

    for i in range(q_recons.shape[1]):

        jcs_pedal = mod_ped.segments["Pedal_left"].frame(q_pedal[:, i])
        jcs_hand = model.segments["hand_left"].frame(q_recons[:, i])

        jcs_hand_T = jcs_hand.T
        jcs_hand_pedal = jcs_hand_T @ jcs_pedal
        #print(jcs_pedal_hand.to_array())
        R = jcs_hand_pedal[:3, :3] # matrice 3×3
        t = jcs_hand_pedal[3,:3]  # vecteur 3×1
        #origin[:,i] = R @ point_app + t #dans le referentiel de la main
        origin[:, i] = jcs_pedal[:3, :3] @ point_app + jcs_pedal[:3, 3] #dans le ref global
        #print(origin)
    force_conca = -np.concatenate((force[0,:,:], force[1,:,:]), axis=0)
    plt.plot(origin[0,:], label='x')
    plt.plot(origin[1, :], label='y')
    plt.plot(origin[2, :], label='z')
    plt.legend()
    plt.show()

    for i in range(q_recons.shape[1]):
        q = q_recons[:,i]
        qdot = qdot_recons[:,i]
        qddot = qddot_recons[:,i]
        model.external_force_set.reset()
        # Proceed with the inverse dynamics
        model.external_force_set.add(segment_name="hand_left", force=force_conca[:,i],
                                      point_of_application = origin[:,i], frame_of_reference= biorbd.ExternalForceSet.Frame.WORLD)  # --> sur le segment, point d'app et force dans le repere global
        tau[:,i] = model.inverse_dynamics(q, qdot, qddot)
        #print(f"Inverse dynamics tau: {tau}")

        dof_name = model.dof_names


    return tau, dof_name

def main():
    # Load a predefined model

    tau, dof_name = inverse_dynamic(model_path, q_path, qdot_path, qddot_path)


    plt.figure()
    for i in range(len(dof_name)):
        plt.plot(tau[i,:], label=dof_name[i])
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()