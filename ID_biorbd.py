from pathlib import Path
import numpy as np
import biorbd
import matplotlib.pyplot as plt
from biorbd import ExternalForceSet
from scipy.signal import find_peaks, butter, filtfilt

ESSAI = "Collecte_18_03"
MODE_PEDALAGE = "concentric"
PUISSANCE = "left"

#
# This examples shows how to
#     1. Load a model
#     2. Position the model at a chosen position (Q), velocity (Qdot) and acceleration (Qddot)
#     3. Compute the generalized forces (tau) at this state (inverse dynamics)
#     4. Print them to the console
#
# Please note that this example will work only with the Eigen backend
#
model_path = f"/Users/leo/Desktop/Projet/{ESSAI}/model_{ESSAI}.bioMod"
model_pedal_path = '/Users/leo/Desktop/Projet/modele_opensim/model_pedal.bioMod'
q_path     = f"/Users/leo/Desktop/Projet/{ESSAI}/{MODE_PEDALAGE}_{PUISSANCE}W/q_inverse_kinematic.npy"
q_pedal_path = f"/Users/leo/Desktop/Projet/{ESSAI}/{MODE_PEDALAGE}_{PUISSANCE}W/inverse_kinematic_pedal.npy"
qdot_path  = f"/Users/leo/Desktop/Projet/{ESSAI}/{MODE_PEDALAGE}_{PUISSANCE}W/qdot_inverse_kinematic.npy"
qddot_path  = f"/Users/leo/Desktop/Projet/{ESSAI}/{MODE_PEDALAGE}_{PUISSANCE}W/qddot_inverse_kinematic.npy"
force_path = f"/Users/leo/Desktop/Projet/{ESSAI}/{MODE_PEDALAGE}_{PUISSANCE}W/constraint_global.npy"
force_pedal_path = f"/Users/leo/Desktop/Projet/{ESSAI}/{MODE_PEDALAGE}_{PUISSANCE}W/constraint_pedal.npy"



def inverse_dynamic(model_path, q_path, qdot_path, qddot_path):
    current_file_dir = Path(__file__).parent
    model = biorbd.Biorbd(model_path)
    force = np.load(force_path)
    #force_pedal = np.load(force_pedal_path)

    nq = model.nb_q
    print("DoF du modèle :", nq)


    q_recons = np.load(q_path)
    #qdot_recons = np.load(qdot_path)
    #qddot_recons = np.load(qddot_path)
    qdot_recons = np.gradient(q_recons, axis=1)

    fs = 100
    cutoff = 6
    dt = 1 / fs
    b, a = butter(4, cutoff / (fs / 2), btype='low')
    qdot_filt = filtfilt(b, a, qdot_recons, axis=1)

    qddot_recons = np.gradient(qdot_filt, axis=1)
    qddot_filt = filtfilt(b, a, qddot_recons, axis=1)
    tau = np.zeros((nq, int(q_recons.shape[1])))

    origin = np.zeros((3, q_recons.shape[1]))
    origin_hand = np.zeros((3, q_recons.shape[1]))

    #print(origin.shape)
    q_pedal = np.load(q_pedal_path)
    mod_ped = biorbd.Biorbd(model_pedal_path)

    force_pedal_hand = np.zeros((3, q_recons.shape[1]))
    moment_pedal_hand = np.zeros((3, q_recons.shape[1]))


    point_app = np.zeros(3) #point d'application dans le repère pédale

    for i in range(q_recons.shape[1]):

        jcs_pedal = mod_ped.segments["Pedal_left"].frame(q_pedal[:, i])
        jcs_hand = model.segments["hand_left"].frame(q_recons[:, i])

        R_hand = jcs_hand[:3,:3]
        t_hand = jcs_hand[:3,3]

        R_pedal = jcs_pedal[:3,:3]
        t_pedal = jcs_pedal[:3,3]

        origin_hand[:,i] = R_hand.T @ (R_pedal @ point_app + t_pedal - t_hand)
        origin[:, i] = jcs_pedal[:3, :3] @ point_app + jcs_pedal[:3, 3] #dans le ref global


        #force_pedal_hand[:,i] = R @ force_pedal[1,:,i]
        #moment_pedal_hand[:,i] = R @ force_pedal[0,:,i] + np.cross(t, force_pedal[1,:,i])
    force_conca = -np.concatenate((force[0, :, :], force[1, :, :]), axis=0)
    #force_pedal_conca = -np.concatenate((force_pedal[0, :, :], force_pedal[1, :, :]), axis=0)


    plt.plot(origin[0,:], label='x')
    plt.plot(origin[1, :], label='y')
    plt.plot(origin[2, :], label='z')
    plt.plot(origin_hand[0, :], label='x_hand')
    plt.plot(origin_hand[1, :], label='y_hand')
    plt.plot(origin_hand[2, :], label='z_hand')
    plt.legend()
    plt.show()

    #------ Derive


    force_pedal_conca = -np.concatenate((moment_pedal_hand, force_pedal_hand), axis=0)
    for i in range(q_recons.shape[1]):
        q = q_recons[:,i]
        qdot = qdot_filt[:,i]
        qddot = qddot_filt[:,i]


        model.external_force_set.reset()
        # Proceed with the inverse dynamics
        model.external_force_set.add(segment_name="hand_left", force=force_conca[:,i],
                                      point_of_application = origin_hand[:,i], frame_of_reference= biorbd.ExternalForceSet.Frame.WORLD)  # --> sur le segment, point d'app et force dans le repere global
        #model.external_force_set.add(segment_name="hand_left", force=force_pedal_conca[:, i],
        #                             point_of_application= origin_hand[:,i], frame_of_reference= biorbd.ExternalForceSet.Frame.LOCAL)  # --> sur le segment, point d'app et force dans le repere global
        #
        tau[:,i] = model.inverse_dynamics(q, qdot, qddot)
        #print(f"Inverse dynamics tau: {tau}")


        dof_name = model.dof_names

    i = 14
    plt.figure()
    plt.plot(qdot_filt[i, :], label="qdot")
    plt.plot(tau[i, :], label="tau")
    plt.legend()
    plt.title("Alignement q / tau")
    plt.show()

    p = tau[i,:] * qdot_filt[i,:]
    print("fraction excentrique (p<0):", (p < 0).mean())
    print("mean power:", p.mean())

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

def plot_segment_grid(
    dof_name,
    peaks_sel,
    tau_list,                  # liste de tau_sel (shape: n_dof x n_frames_sel)
    labels=None,               # ex: ["RGBD-based", "redundant-Vicon-based", "minimal-Vicon-based"]
    colors=None,               # ex: ["royalblue", "tomato", "seagreen"]
    layout=None,
    n_points=200,
    ylabel="Torque (N·m)",
):
    if layout is None:
        raise ValueError("layout est requis (dict segment -> liste de (dof, titre)).")

    if labels is None:
        labels = [f"method {i+1}" for i in range(len(tau_list))]
    if colors is None:
        colors = [None] * len(tau_list)  # laisse matplotlib choisir

    # Prépare figure: nb colonnes = max nb de DoF sur une ligne
    segments = list(layout.keys())
    n_rows = len(segments)
    n_cols = max(len(layout[s]) for s in segments)

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(11.69, 8.27),
        sharex=True
    )
    plt.subplots_adjust(
        left=0.07,
        right=0.98,
        top=0.93,
        bottom=0.08,
        wspace=0.25,
        hspace=0.35
    )
    if n_rows == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes[:, None]

    x = np.linspace(0, 100, n_points)

    for r, seg in enumerate(segments):
        dofs = layout[seg]
        for c in range(n_cols):
            ax = axes[r, c]

            # case vide si la ligne a moins de colonnes
            if c >= len(dofs):
                ax.axis("off")
                continue

            dof, title = dofs[c]
            if dof not in dof_name:
                ax.set_title(f"{title}\n(MISSING: {dof})", fontsize=12)
                ax.axis("off")
                continue

            idx = dof_name.index(dof)

            # Trace toutes les méthodes
            for tau_sel, lab, col in zip(tau_list, labels, colors):
                cyc = extract_cycles_generic(tau_sel[idx, :], peaks_sel)  # (n_cycles, n_points)
                mean_ = np.mean(cyc, axis=0)
                std_  = np.std(cyc, axis=0)

                ax.plot(x, mean_, lw=2, label=lab, color=col)
                ax.fill_between(x, mean_ - std_, mean_ + std_, alpha=0.15, color=col)

            ax.set_title(title, fontsize=13)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=10)

            # Y label seulement sur 1ère colonne de chaque ligne (comme sur ton image)
            if c == 0:
                ax.set_ylabel(f"{seg}\n{ylabel}")
            else:
                ax.set_ylabel("")

            # X label seulement sur dernière ligne
            if r == n_rows - 1:
                ax.set_xlabel("Mean cycle (%)")

    # Une seule légende globale (en haut à droite, comme ton exemple)
    handles, leg_labels = axes[0, 0].get_legend_handles_labels()
    if len(handles) > 0:
        fig.legend(handles, leg_labels, loc="upper right", frameon=False)

    plt.tight_layout(rect=[0, 0, 0.95, 1])  # laisse de la place à la légende
    plt.show()


def eccentric_index_per_cycle(tau, qdot, peaks, dof_name):
    """
    Calcule un indice excentrique par cycle et par DoF
    basé sur la puissance articulaire P = tau * qdot

    Retour :
        results_dict[dof] = {
            "mean_index": ...,
            "std_index": ...,
            "mean_power": ...,
            "ecc_fraction": ...
        }
    """

    n_dof = tau.shape[0]
    results = {}
    for d in range(6,n_dof):

        cycle_indices = []

        for i in range(len(peaks) - 1):
            i0, i1 = peaks[i], peaks[i + 1]

            tau_seg = tau[d, i0:i1]
            qdot_seg = qdot[d, i0:i1]

            power = tau_seg * qdot_seg

            if np.sum(np.abs(power)) == 0:
                continue

            ecc_energy = np.sum(np.abs(power[power < 0]))
            total_energy = np.sum(np.abs(power))

            ecc_index = ecc_energy / total_energy
            cycle_indices.append(ecc_index)

        if len(cycle_indices) == 0:
            continue

        cycle_indices = np.array(cycle_indices)

        results[dof_name[d]] = {
            "mean_index": np.mean(cycle_indices),
            "std_index": np.std(cycle_indices),
            "mean_power": np.mean(tau[d, :] * qdot[d, :]),
            "ecc_fraction_frames": np.mean((tau[d, :] * qdot[d, :]) < 0)
        }

    return results


def main():
    # Load a predefined model


    tau, dof_name = inverse_dynamic(model_path, q_path, qdot_path, qddot_path)

    np.save(f"/Users/leo/Desktop/Projet/{ESSAI}/{MODE_PEDALAGE}_{PUISSANCE}W/tau_inverse_dynamic", tau)
    plt.figure()
    for i in range(len(dof_name)):
        plt.plot(tau[i,500:], label=dof_name[i])
    plt.legend()
    plt.show()

    # ----------- Paramètres utilisateur -----------
    if ESSAI == "Collecte_25_11":
        if MODE_PEDALAGE == "concentric": #vérifier les frames d'initialisations
            if PUISSANCE == "40":
                START = 2000  # frame de début (ex : 2000)
                END = 5200  # frame de fin
            elif PUISSANCE == "60":
                START = 2000  # frame de début (ex : 2000)
                END = 5000  # frame de fin
            elif PUISSANCE == "80":
                START = 1500  # frame de début (ex : 2000)
                END = 4000  # frame de fin
            else:
                print("PB PUISSANCE")
        elif MODE_PEDALAGE == "eccentric":
            if PUISSANCE == "40":
                START = 2000  # frame de début (ex : 2000)
                END = 5000  # frame de fin
            elif PUISSANCE == "60":
                START = 1500  # frame de début (ex : 2000)
                END = 3500  # frame de fin
            elif PUISSANCE == "80":
                START = 7000  # frame de début (ex : 2000)
                END = 10000  # frame de fin
            else:
                print("PB PUISSANCE")
    elif ESSAI == "Collecte_13_03":
        START = 2300  # frame de début (ex : 2000)
        END = 3400  # frame de fin

    elif ESSAI == "Collecte_18_03":
        START = 1000  # frame de début (ex : 2000)
        END = 4000  # frame de fin

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

    q = np.load(q_path)
    # Signal de référence
    ref_signal = q[ref_idx, :]  # ou q_recons[ref_idx,:] si nécessaire

    # Sélection plage temporelle
    ref_signal_sel = ref_signal[START:END]

    # Détection des peaks
    peaks_sel, _ = find_peaks(ref_signal_sel, distance=100)

    print("Nombre de cycles détectés :", len(peaks_sel) - 1)

    #====== Vérif mode de pédalage

    # Charger qdot filtré pour la même fenêtre
    qdot_full = np.load(qdot_path)
    qdot_sel = qdot_full[:, START:END]

    ecc_results = eccentric_index_per_cycle(
        tau_sel,
        qdot_sel,
        peaks_sel,
        dof_name
    )
    print("\n================ INDICE EXCENTRIQUE PAR DOF ================\n")

    print(f"{'DoF':60s} | {'Ecc Index':>10s} | {'Std':>8s} | {'% Frames P<0':>15s}")
    print("-" * 100)

    for dof, vals in ecc_results.items():
        print(f"{dof:60s} | "
              f"{vals['mean_index']:.3f}      | "
              f"{vals['std_index']:.3f}   | "
              f"{100 * vals['ecc_fraction_frames']:.1f} %")

    # ==========================================================
    # === SUBPLOT : COUPLES / FORCES PAR DOF SUR LE CYCLE ===
    # =========================================================

    # ---- Layout : 1 ligne = 1 segment ; chaque élément = (dof_name_exact, "titre subplot")
    LAYOUT = {
        "Clavicle": [
            ("thorax_offset_sternoclavicular_left_r1_RotX", "Pro/retraction"),
            ("thorax_offset_sternoclavicular_left_r2_RotY", "Depression/Elevation"),
        ],
        "Scapula": [
            ("scapula_left_rotation_transform_RotX", "Pro/retraction"),
            ("scapula_left_rotation_transform_RotY", "Lat/med rotation"),
            ("scapula_left_rotation_transform_RotZ", "Tilt"),
        ],
        "Humerus": [
            ("scapula_left_offset_shoulder_left_plane_RotX", "Plane of elevation"),
            ("scapula_left_offset_shoulder_left_ele_RotY", "Elevation"),
            ("scapula_left_offset_shoulder_left_rotation_RotZ", "Axial rotation"),
        ],
        "Forearm": [
            ("humerus_left_offset_elbow_left_flexion_RotZ", "Flexion/extension"),
            ("ulna_left_offset_pro_sup_left_RotY", "Pronation/supination"),
        ],
    }

    plot_segment_grid(
        dof_name=dof_name,
        peaks_sel=peaks_sel,
        tau_list=[tau_sel],  # <- une seule méthode pour l'instant
        labels=[MODE_PEDALAGE + " " + PUISSANCE + " W"],
        colors=["royalblue"],
        layout=LAYOUT,
        n_points=200,
        ylabel="Torque (N·m)"
    )



if __name__ == "__main__":
    main()