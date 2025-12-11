from pathlib import Path
import numpy as np
import biorbd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, butter, filtfilt

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
force_pedal_path = "/Users/leo/Desktop/Projet/Collecte_25_11/IK/constraint_pedal_40W.npy"



def inverse_dynamic(model_path, q_path, qdot_path, qddot_path):
    current_file_dir = Path(__file__).parent
    model = biorbd.Biorbd(model_path)
    force = np.load(force_path)

    nq = model.nb_q
    print("DoF du modèle :", nq)


    q_recons = np.load(q_path)
    qdot_recons = np.load(qdot_path)

    fs = 100
    cutoff = 6
    dt = 1 / fs
    b, a = butter(4, cutoff / (fs / 2), btype='low')
    qdot_filt = filtfilt(b, a, qdot_recons, axis=1)

    qddot_recons = np.load(qddot_path)
    qddot_filt = filtfilt(b, a, qddot_recons, axis=1)
    tau = np.zeros((nq, int(q_recons.shape[1])))

    origin = np.zeros((3, q_recons.shape[1]))
    #print(origin.shape)
    q_pedal = np.load(q_pedal_path)
    mod_ped = biorbd.Biorbd(model_pedal_path)

    point_app = np.zeros(3) #point d'application dans le repère pédale

    for i in range(q_recons.shape[1]):

        jcs_pedal = mod_ped.segments["Pedal_left"].frame(q_pedal[:, i])
        jcs_hand = model.segments["hand_left"].frame(q_recons[:, i])
        jcs_hand_pedal = jcs_hand.T @ jcs_pedal
        #print(jcs_pedal_hand.to_array())
        R = jcs_hand_pedal[:3, :3] # matrice 3×3
        t = jcs_hand_pedal[3,:3]  # vecteur 3×1
        #origin[:,i] = R @ point_app + t #dans le referentiel de la main
        origin[:, i] = jcs_pedal[:3, :3] @ point_app + jcs_pedal[:3, 3] #dans le ref global
        #print(origin)

        #force_pedal_hand[:,i] = R @ force_pedal[1,:,i]
        #moment_pedal_hand[:,i] = R @ force_pedal[0,:,i] + np.cross(t, force_pedal[1,:,i])
    force_conca = -np.concatenate((force[0, :, :], force[1, :, :]), axis=0)

    plt.plot(origin[0,:], label='x')
    plt.plot(origin[1, :], label='y')
    plt.plot(origin[2, :], label='z')
    plt.legend()
    plt.show()

    #------ Derive


    #force_pedal_conca = -np.concatenate((moment_pedal_hand, force_pedal_hand), axis=0)
    for i in range(q_recons.shape[1]):
        q = q_recons[:,i]
        qdot = qdot_filt[:,i]
        qddot = qddot_filt[:,i]


        model.external_force_set.reset()
        # Proceed with the inverse dynamics
        model.external_force_set.add(segment_name="hand_left", force=force_conca[:,i],
                                      point_of_application = origin[:,i], frame_of_reference= biorbd.ExternalForceSet.Frame.WORLD)  # --> sur le segment, point d'app et force dans le repere global
        #model.external_force_set.add(segment_name="hand_left", force=force_pedal_conca[:, i],
        #                             point_of_application=origin[:, i],
        #                             frame_of_reference=biorbd.ExternalForceSet.Frame.LOCAL)  # --> sur le segment, point d'app et force dans le repere global

        tau[:,i] = model.inverse_dynamics(q, qdot, qddot)
        #print(f"Inverse dynamics tau: {tau}")

        dof_name = model.dof_names


    return tau, dof_name

def extract_cycles_generic(signal, peaks):
    out = []
    for i in range(len(peaks) - 1):
        seg = signal[peaks[i]:peaks[i + 1]]
        seg_norm = np.interp(
            np.linspace(0, 1, 200),
            np.linspace(0, 1, len(seg)),
            seg
        )
        out.append(seg_norm)
    return np.array(out)

def main():
    # Load a predefined model


    tau, dof_name = inverse_dynamic(model_path, q_path, qdot_path, qddot_path)

    np.save("/Users/leo/Desktop/Projet/Collecte_25_11/ID/tau_inverse_dynamic_Sidonie_40w", tau)
    plt.figure()
    for i in range(len(dof_name)):
        plt.plot(tau[i,500:], label=dof_name[i])
    plt.legend()
    plt.show()

    # ----------- Paramètres utilisateur -----------
    print(dof_name)
    DOF_TO_PLOT = ['thorax_rotation_transform_RotX',
                   'thorax_rotation_transform_RotY',
                   'thorax_rotation_transform_RotZ',
                   'thorax_offset_sternoclavicular_left_r1_RotX',
                   'thorax_offset_sternoclavicular_left_r2_RotY',
                   'scapula_left_rotation_transform_RotX',
                   'scapula_left_rotation_transform_RotY',
                   'scapula_left_rotation_transform_RotZ',
                   'scapula_left_offset_shoulder_left_plane_RotX',
                   'scapula_left_offset_shoulder_left_ele_RotY',
                   'scapula_left_offset_shoulder_left_rotation_RotZ',
                   'humerus_left_offset_elbow_left_flexion_RotZ',
                   'ulna_left_offset_pro_sup_left_RotY',
                   'hand_left_rotation_transform_RotX',
                   'hand_left_rotation_transform_RotZ'
                   ]  # soit "ALL", soit ["dof1", "dof2", ...]
    START = 3000  # frame de début (ex : 2000)
    END = 5000  # frame de fin

    # ----------- Sélection plage temporelle --------
    tau_sel = tau[:, START:END]
    # ==========================================================
    # DÉTECTION DES CYCLES À PARTIR D’UN DOF DE RÉFÉRENCE
    # ==========================================================

    # Choix automatique d’un DOF de référence pour détecter les cycles
    ref_idx = None

    # Cherche un DoF du coude
    for i, name in enumerate(dof_name):
        if "elbow" in name.lower():
            ref_idx = i
            break

    # Sinon cherche un DoF de l'épaule
    if ref_idx is None:
        for i, name in enumerate(dof_name):
            if "shoulder" in name.lower():
                ref_idx = i
                break

    # Sinon prend le DoF 0
    if ref_idx is None:
        ref_idx = 0

    print(f"DoF utilisé comme référence du cycle : {dof_name[ref_idx]}")

    # Signal de référence
    ref_signal = tau[ref_idx, :]  # ou q_recons[ref_idx,:] si nécessaire

    # Sélection plage temporelle
    ref_signal_sel = ref_signal[START:END]

    # Détection des peaks
    peaks_sel, _ = find_peaks(ref_signal_sel, distance=100)

    print("Nombre de cycles détectés :", len(peaks_sel) - 1)
    # ==========================================================
    # === SUBPLOT : COUPLES / FORCES PAR DOF SUR LE CYCLE ===
    # =========================================================


    # ----------- Sélection DoF à tracer ------------
    if DOF_TO_PLOT == "ALL":
        selected_dofs = dof_name
    else:
        selected_dofs = DOF_TO_PLOT

    # ----------- Construction cycles τ par DoF --------
    cycles_tau = {}
    mean_tau = {}
    std_tau = {}

    for dof in selected_dofs:
        idx = dof_name.index(dof)
        cyc = extract_cycles_generic(tau_sel[idx, :], peaks_sel)
        cycles_tau[dof] = cyc
        mean_tau[dof] = np.mean(cyc, axis=0)
        std_tau[dof] = np.std(cyc, axis=0)

    # ----------- Plot final : GRILLE DE SUBPLOTS -------------------------

    import math

    x = np.linspace(0, 100, 200)
    n_dofs = len(selected_dofs)

    # Définition automatique d'une grille
    n_cols = math.ceil(math.sqrt(n_dofs))
    n_rows = math.ceil(n_dofs / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=True)
    axes = axes.flatten()  # pour parcourir facilement

    for ax, dof in zip(axes, selected_dofs):

        # cycles individuels
        for c in cycles_tau[dof]:
            ax.plot(x, c, color="gray", alpha=0.25)

        # moyenne
        ax.plot(x, mean_tau[dof], linewidth=2, color="blue")

        # écart-type
        ax.fill_between(
            x,
            mean_tau[dof] - std_tau[dof],
            mean_tau[dof] + std_tau[dof],
            color="blue",
            alpha=0.15
        )

        ax.set_title(dof, fontsize=8)
        ax.set_ylabel("τ (N·m)")
        ax.grid(True)

    # Supprimer les axes inutilisés si la grille est trop grande
    for i in range(len(selected_dofs), len(axes)):
        fig.delaxes(axes[i])

    # Label global
    plt.xlabel("Cycle (%)")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()