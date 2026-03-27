from pathlib import Path
from math import pi
import numpy as np
import matplotlib.pyplot as plt
import ezc3d
import biorbd
import csv
from scipy.signal import find_peaks
from pyomeca import Analogs

try:
    import bioviz

    biorbd_viz_found = False
except ModuleNotFoundError:
    biorbd_viz_found = False


#Collecte_25_11

ESSAI = "Collecte_18_03"
MODE_PEDALAGE = "eccentric"
PUISSANCE = "left"


# === Choix des frames à analyser ===
END_FRAME   = None    # Dernière frame (None = dernière frame du fichier)


# === 1. Markers du modèle, DANS L'ORDRE DU .bioMod ===
MODEL_MARKERS = [
    "Crank_Axe_L",
    "Crank_Axe_R",
    "Pedal4",
    "Pedal5",
    "Pedal6",
]

def afficher_entetes_ezc3d(fichier):
    c3d = ezc3d.c3d(str(fichier))

    # Paramètres ANALOG
    params = c3d["parameters"]

    # Vérification de la présence de LABELS
    if "ANALOG" in params and "LABELS" in params["ANALOG"]:
        labels = params["ANALOG"]["LABELS"]["value"]
        print("Liste des canaux analogiques :")
        for i, label in enumerate(labels):
            print(f"  {i + 1}. {label}")
        return labels

    else:
        print("⚠️  Pas de LABELS trouvés dans la section ANALOG.")
        print("Clés disponibles :", params.get("ANALOG", {}).keys())


def find_trigger(file):
    # Charger le canal analogique
    analog = Analogs.from_c3d(filename=file, usecols=['Electric Resistance.1']).values[0]

    # Charger le c3d
    c3d = ezc3d.c3d(file)

    # Lire les fréquences
    analog_rate = c3d["parameters"]["ANALOG"]["RATE"]["value"][0]  # ex: 2000
    point_rate = c3d["parameters"]["POINT"]["RATE"]["value"][0]  # ex: 100

    # ratio entre analog et markers
    ratio = int(analog_rate / point_rate)  # ex: 20

    # Trouver les indices (en samples analogiques) où le signal dépasse 2V
    trigger_samples = np.where(analog > 2.0)[0]

    # Si rien ne dépasse → on renvoie 0
    if trigger_samples.size == 0:
        return 0

    # Premier sample dépassant 2V
    first_trigger_sample = trigger_samples[0]

    # Convertir en frame markers
    trigger_frame = first_trigger_sample // ratio

    print("Trigger (sample analog) =", first_trigger_sample)
    print("Trigger (frame marker)  =", trigger_frame)

    return trigger_frame

def build_marker_mapping(c3d_labels):
    mapping = {}
    for name in MODEL_MARKERS:
        if name not in c3d_labels:
            raise ValueError(f"Le marqueur {name} du modèle est absent du C3D.")
        mapping[name] = c3d_labels.index(name)
    return mapping


def extract_relevant_markers(raw_markers, mapping):
    indices = [mapping[name] for name in MODEL_MARKERS]
    return raw_markers[:, indices, :]


def transform_forces_to_global(model, q_recons, F_local, M_local, angle_local, F_crank, M_crank,
                               fs_high=250, fs_low=100, mode="nearest"):
    """
    Transforme les forces/moments en coordonnées globales, en recalent les signaux 250 Hz sur 100 Hz.

    Parameters
    ----------
    model : biorbd.Model
    q_recons : ndarray (DoF x n_frames)
    F_local : ndarray (3 x n_samples_high)
    M_local : ndarray (3 x n_samples_high)
    fs_high : fréquence du signal pédales (ex: 250 Hz)
    fs_low : fréquence des frames cinématiques (ex: 100 Hz)
    mode : "interp" ou "nearest"

    Returns
    -------
    F_global, M_global : ndarray (3 x n_frames)
    """

    n_frames = q_recons.shape[1]

    # ----------------------------
    # 1) Création des timelines
    # ----------------------------
    t_low = np.arange(n_frames) / fs_low                  # temps 100 Hz
    n_high = F_local.shape[1]
    t_high = np.arange(n_high) / fs_high                 # temps 250 Hz

    # ----------------------------
    # 2) Recalage 250 Hz → 100 Hz
    # ----------------------------
    if mode == "interp":
        # interpolation linéaire
        Fp_resampled = np.vstack([
            np.interp(t_low, t_high, F_local[i, :])
            for i in range(3)
        ])
        Mp_resampled = np.vstack([
            np.interp(t_low, t_high, M_local[i, :])
            for i in range(3)
        ])
        angle_resampled = np.vstack([
            np.interp(t_low, t_high, angle_local[i])
            for i in range(3)
        ])

    elif mode == "nearest":
        # index du point 250 Hz le plus proche
        idx = np.searchsorted(t_high, t_low)
        idx = np.clip(idx, 1, len(t_high)-1)

        left = t_high[idx - 1]
        right = t_high[idx]
        choose_right = (right - t_low) < (t_low - left)
        nearest_idx = idx.copy()
        nearest_idx[~choose_right] -= 1

        Fp_resampled = F_local[:, nearest_idx]
        Mp_resampled = M_local[:, nearest_idx]
        angle_resampled = angle_local[nearest_idx]
        Fc_resampled = F_crank[:, nearest_idx]
        Mc_resampled = M_crank[:, nearest_idx]

    else:
        raise ValueError("mode must be 'interp' or 'nearest'")

    angle_resampled = np.mod(angle_resampled-np.pi, 2*np.pi) #passage angle pédalier droit à gauche

    np.save(f"/Users/leo/Desktop/Projet/{ESSAI}/{MODE_PEDALAGE}_{PUISSANCE}W/constraint_crank.npy", [Mc_resampled, Fc_resampled])
    #np.save(f"/Users/leo/Desktop/Projet/{ESSAI}/{MODE_PEDALAGE}_{PUISSANCE}W/crank_angle.npy", angle_resampled)
    # ----------------------------
    # 3) Transformation en global
    # ----------------------------
    F_global = np.zeros((3, n_frames))
    M_global = np.zeros((3, n_frames))


    for i in range(n_frames):
        # force/moment 100 Hz correspondants
        Fp = Fp_resampled[:, i]
        Mp = Mp_resampled[:, i]
        q = q_recons[:, i]

        model.update_kinematics(q)

        T = model.segments["Pedal_left"].frame()

        R = T[:3, :3]
        p = T[:3, 3]

        # transformation
        Fg = R @ Fp
        Mg = R @ Mp + np.cross(p, Fg)

        F_global[:, i] = Fg
        M_global[:, i] = Mg

    return F_global, M_global, angle_resampled

def compute_pedal_angle_from_ground(model, q_recons, unwrap=False):
    """
    Angle de la pédale gauche autour de l’axe Y du ground.
    Le zéro correspond à l’orientation du repère ground.
    """

    n_frames = q_recons.shape[1]
    theta = np.zeros(n_frames)

    for i in range(n_frames):

        RT_ground = model.segments["ground"].frame(q_recons[:, i])

        RT_pedal = model.segments["Pedal_left"].frame(q_recons[:, i])
        RT_ground_ped = np.linalg.inv(RT_ground) @ RT_pedal

        p = RT_ground_ped[:3, 3]

        # axe Y local de la pédale exprimé dans le global
        y = p[1]
        z = p[2]

        # Rotation autour de X → projection dans plan YZ
        angle = np.arctan2(z, y)

        # Bornage dans [0, 2π]
        theta[i] = -angle #% (2 * np.pi)

    if unwrap:
        theta = np.unwrap(theta)

    if ESSAI == "Collecte_18_03" :
        theta = (2*np.pi)-theta


    return theta



def main(show=True):

    model_path = Path("/Users/leo/Desktop/Projet/modele_opensim/model_pedal.bioMod")
    c3d_path = Path(f"/Users/leo/Desktop/Projet/{ESSAI}/C3D_labelled/{MODE_PEDALAGE}_{PUISSANCE}W.c3d")
    sensix_path = Path(f"/Users/leo/Desktop/Projet/{ESSAI}/pedales/Results-{MODE_PEDALAGE}_{PUISSANCE}w_001.lvm")


    #afficher_entetes_ezc3d(str(c3d_path))

    model = biorbd.Biorbd(str(model_path))
    nq = model.nb_q

    print("DoF du modèle :", nq)

    c3d = ezc3d.c3d(str(c3d_path))
    raw_markers = c3d["data"]["points"][:3, :, :]
    labels = c3d["parameters"]["POINT"]["LABELS"]["value"]
    units = c3d["parameters"]["POINT"]["UNITS"]["value"][0]

    if units.lower().startswith("mm"):
        raw_markers /= 1000.0

    mapping = build_marker_mapping(labels)
    markers = extract_relevant_markers(raw_markers, mapping)

    # === APPLY FRAME SELECTION HERE ===
    markers = markers[:, :, find_trigger(str(c3d_path)):END_FRAME] #find_trigger(str(c3d_path))
    n_frames = markers.shape[2]

    markers = markers.transpose(2, 0, 1)  # => (n_frames, 3, nbDoF)

    markers = [frame for frame in markers]  # => liste de matrices


    q_recons = np.zeros((nq, n_frames))

    kalman = biorbd.ExtendedKalmanFilterMarkers(model, frequency=100)

    if ESSAI == "Collecte_25_11":
        if MODE_PEDALAGE == "concentric": #vérifier les frames d'initialisations
            if PUISSANCE == "40":
                init = 4000
                dephasage = 0 #frame retard velo
            elif PUISSANCE == "60":
                init = 3000
                dephasage = 0
            elif PUISSANCE == "80":
                init = 3000
                dephasage = 0
            else:
                print("PB PUISSANCE")
        elif MODE_PEDALAGE == "eccentric":
            if PUISSANCE == "40":
                init = 2000
                dephasage = 0
            elif PUISSANCE == "60":
                init = 3000
                dephasage = 0
            elif PUISSANCE == "80":
                init = 8000
                dephasage = 0
            else:
                print("PB PUISSANCE")
        else:
            print("PB MODE PEDALAGE")
        print("Filtre de Kalman initialisé à la frame", init)

    elif ESSAI == "Collecte_13_03":
        init = 1000
        dephasage = 0

    elif ESSAI == "Collecte_18_03":
        init = 1000
        dephasage = 0
    q_i, _, _ = kalman.reconstruct_frame(markers[init])
    for i, (q_i, _, _) in enumerate(kalman.reconstruct_frames(markers)):
        q_recons[:, i] = q_i
        # qdot_recons[:, i] = qdot_i
        # qddot_recons[:, i] = qddot_i

        if i % 200 == 0:
            print(f"Frame {i}/{n_frames}")

    q_recons=q_recons[:,dephasage:]

    print("IK terminé.")

    np.save(f"/Users/leo/Desktop/Projet/{ESSAI}/{MODE_PEDALAGE}_{PUISSANCE}W/inverse_kinematic_pedal.npy", q_recons)
    print("données IK enregistrées :)")



    all_data = []
    with open(sensix_path, 'r') as f:
        csvreader = csv.reader(f, delimiter='\n')
        for row in csvreader:
            all_data.append(np.array(row[0].split("\t")))
    all_data = np.array(all_data, dtype=float).T
    plt.plot(all_data[1,:], label='Fx')
    plt.plot(all_data[2,:], label='Fy')
    plt.plot(all_data[3,:], label='Fz')
    plt.legend()
    plt.show()

    plt.plot(all_data[4, :], label='Mx')
    plt.plot(all_data[5, :], label='My')
    plt.plot(all_data[6, :], label='Mz')
    plt.legend()
    plt.show()

    global_force, global_moment, crank_angle = transform_forces_to_global(model, q_recons, all_data[1:4,:], all_data[4:7,:], all_data[19,:],all_data[21:24,:], all_data[24:27,:])
    global_constraint = [global_moment, global_force]
    np.save(f"/Users/leo/Desktop/Projet/{ESSAI}/{MODE_PEDALAGE}_{PUISSANCE}W/constraint_global.npy", global_constraint)

    print("Forces et Moments enregistrés")
    print("markers frames:", n_frames)
    print("forces frames:", all_data.shape[1])  # total forces
    print("IK frames    :", q_recons.shape[1])

    plt.plot(global_force[0,:], label='Fx')
    plt.plot(global_force[1,:], label='Fy')
    plt.plot(global_force[2,:], label='Fz')
    plt.legend()
    plt.show()

    plt.plot(global_moment[0, :], label='Mx')
    plt.plot(global_moment[1, :], label='My')
    plt.plot(global_moment[2, :], label='Mz')
    plt.legend()
    plt.show()
    if show and biorbd_viz_found:
        modelviz = biorbd.Model(str(model_path))
        b = bioviz.Viz(loaded_model=modelviz)
        b.load_movement(q_recons)
        b.exec()

    theta = compute_pedal_angle_from_ground(model, q_recons)
    diff = crank_angle[100]-theta[100] #3,28 rad
    print(diff)
    theta = (theta + diff) % (2*np.pi)
    plt.plot(theta, label = "angle  recalculé")
    plt.plot(crank_angle, label = "angle vélo")
    plt.title("angle pédale / pédalier")
    plt.xlabel("Frame")
    plt.ylabel("angle (rad")
    plt.legend()
    plt.show()

    #np.save(f"/Users/leo/Desktop/Projet/{ESSAI}/{MODE_PEDALAGE}_{PUISSANCE}W/crank_angle.npy", crank_angle)
    np.save(f"/Users/leo/Desktop/Projet/{ESSAI}/{MODE_PEDALAGE}_{PUISSANCE}W/crank_angle.npy", theta)


if __name__ == "__main__":
    main(show=True)
