from pathlib import Path
from math import pi
from pyomeca import Analogs
import numpy as np
import matplotlib.pyplot as plt
import ezc3d
import biorbd
from scipy.signal import find_peaks
try:
    import bioviz

    biorbd_viz_found = True
except ModuleNotFoundError:
    biorbd_viz_found = False


# === Choix des frames à analyser ===
END_FRAME   = 8000    # Dernière frame (None = dernière frame du fichier)


# === 1. Markers du modèle, DANS L'ORDRE DU .bioMod ===
MODEL_MARKERS = [
    "Ster",
    "Xiph",
    "C7",
    "T10",
    "Clav_SC",
    "Clav_AC",
    "Clav_Mid",
    "Scap_AA",
    "Scap_TS",
    "Scap_IA",
    "Delt",
    "EpicI",
    "EpicM",
    "ArmI",
    "Elbow",
    "StylU",
    "LArmI",
    "StylR",
    "Index_Base",
    "Little_Base",
    "Hand_Top",
]

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


def numpy_markers_to_nodes(markers_frame):
    nodes = []
    for i in range(markers_frame.shape[1]):
        x, y, z = markers_frame[:, i]
        node = biorbd.NodeSegment(float(x), float(y), float(z))
        nodes.append(node)
    return nodes


def main(show=True):

    model_path = Path("/Users/leo/Desktop/Projet/modele_opensim/wu_bras_gauche_seth_left_Sidonie.bioMod")
    c3d_path = Path("/Users/leo/Desktop/Projet/Collecte_25_11/C3D_labelled/concentric_40W.c3d")

    model = biorbd.Model(str(model_path))
    nq = model.nbQ()

    c3d = ezc3d.c3d(str(c3d_path))
    raw_markers = c3d["data"]["points"][:3, :, :]
    labels = c3d["parameters"]["POINT"]["LABELS"]["value"]
    units = c3d["parameters"]["POINT"]["UNITS"]["value"][0]

    if units.lower().startswith("mm"):
        raw_markers /= 1000.0

    mapping = build_marker_mapping(labels)
    markers = extract_relevant_markers(raw_markers, mapping)

    # === APPLY FRAME SELECTION HERE ===
    markers = markers[:, :, find_trigger(str(c3d_path)):END_FRAME]
    n_frames = markers.shape[2]

    kalman = biorbd.KalmanReconsMarkers(model)
    q_recons = np.zeros((nq, n_frames))
    marker_error = np.zeros((nq, n_frames))


    for i in range(n_frames):
        marker_nodes = numpy_markers_to_nodes(markers[:, :, i])

        q = biorbd.GeneralizedCoordinates(model)
        qdot = biorbd.GeneralizedVelocity(model)
        qddot = biorbd.GeneralizedAcceleration(model)

        kalman.reconstructFrame(model, marker_nodes, q, qdot, qddot)

        #markers_model = model.markers(q)  # liste de nodes
        #markers_model = np.array([m.to_array() for m in markers_model]).T  # shape (3, nMarkers)
        #errors = np.linalg.norm(markers_model - markers[:, :, i], axis=0)  # en mètres
        #marker_error[:, i] = errors

        q_recons[:, i] = q.to_array()

        if i % 200 == 0:
            print(f"Frame {i}/{n_frames}")



    print("IK terminé.")
    marker_error_mm = marker_error * 1000

    plt.plot(np.mean(marker_error_mm, axis=1))
    plt.xlabel("Frame")
    plt.ylabel("Erreur marker moyenne (mm)")
    plt.show()

    # === Extraction coude ===
    # ===========================
    # CHOIX DES DOF À ANALYSER
    # ===========================

    JOINTS = {
        "abduction épaule": 0,
        "Flexion de l'épaule": 1,
        "Rot axiale de l'épaule": 2
    }

    # ===========================
    # 1) Détection des pics via le coude (référence du cycle)
    # ===========================
    coude = np.unwrap(q_recons[14, :])


    plt.plot(np.rad2deg(q_recons[11, :]), label="adbuction epaule kalmann")  #--> Abduction épaule
    plt.plot(np.rad2deg(q_recons[12, :]), label="Flexion epaule kalmann")  #--> Flexion épaule
    plt.plot(np.rad2deg(q_recons[13, :]), label="Rot axiale epaule kalmann")  #-->
    plt.plot(np.rad2deg(coude), label="Coude kalmann")  #--> Flexion coude

    # ===========================
    # 2) Enregistrement des données
    # ===========================
    np.save("/Users/leo/Desktop/Projet/Collecte_25_11/IK/inverse_kinematic_sidonie_40W.npy", q_recons)
    print("données IK enregistrées :)")

    # ===========================
    # 3) Matrice de rot
    # ===========================
    def wrap_to_180(angle_deg):
        return (np.unwrap(angle_deg) + 180) % 360 - 180

    n_frames = q_recons.shape[1]

    shoulder_euler = np.zeros((3, n_frames))  # plan / élévation / rotation
    elbow_euler = np.zeros((3, n_frames))  # suivant "XYZ"

    for i in range(n_frames):
        q_i = biorbd.GeneralizedCoordinates(q_recons[:, i])

        rt_scap = model.globalJCS(q_i, "scapula_left")
        rt_humerus = model.globalJCS(q_i, "humerus_left")
        rt_ulna = model.globalJCS(q_i, "ulna_left")

        rt_scap_T = rt_scap.transpose()
        rt_scap_hum = rt_scap_T.multiply(rt_humerus)
        rt_hum_T = rt_humerus.transpose()
        rt_hum_ulna = rt_hum_T.multiply(rt_ulna)

        ang_sh = biorbd.RotoTrans.toEulerAngles(rt_scap_hum, "xyz")
        ang_el = biorbd.RotoTrans.toEulerAngles(rt_hum_ulna, "xyz")

        shoulder_euler[:, i] = np.degrees(ang_sh.to_array())
        elbow_euler[:, i] = np.degrees(ang_el.to_array())
        print(i)

    # Wrapping dans [-180, 180]
    shoulder_euler = wrap_to_180(shoulder_euler)
    elbow_euler = wrap_to_180(elbow_euler)

    plt.plot(shoulder_euler[0, :], label="Épaule – plan d'élévation")
    plt.plot(shoulder_euler[1, :], label="Épaule – élévation")
    plt.plot(shoulder_euler[2, :], label="Épaule – rotation axiale")
    #plt.plot(elbow_euler[2, :], label="Coude – rotation flexion/extension")

    plt.legend()


    # ===========================
    # 4) Extraction & normalisation des cycles
    # ===========================
    ref_signal = np.degrees(elbow_euler[2, :])  # --> flexion coud
    peaks, _ = find_peaks(ref_signal, distance=100)

    print("Nombre de cycles détectés :", len(peaks) - 1)

    def extract_cycles(signal_deg, peaks):
        cycles = []
        for i in range(len(peaks) - 1):
            cyc = signal_deg[peaks[i]:peaks[i + 1]]
            cyc_norm = np.interp(
                np.linspace(0, 1, 200),
                np.linspace(0, 1, len(cyc)),
                cyc
            )
            cycles.append(cyc_norm)
        return np.array(cycles)

    # Extraire les cycles pour chaque articulation
    cycles_per_joint = {}
    mean_per_joint = {}
    std_per_joint = {}

    for name, dof in JOINTS.items():
        #sig = np.degrees(q_recons[dof, :])
        cycles = extract_cycles(shoulder_euler[dof,:], peaks)
        cycles_per_joint[name] = cycles
        mean_per_joint[name] = np.mean(cycles, axis=0)
        std_per_joint[name] = np.std(cycles, axis=0)

    cycles = extract_cycles(elbow_euler[2, :], peaks)

    #On ajoute ke coude qui est dans un autre np
    cycles_per_joint["flexion coude"] = cycles
    mean_per_joint["flexion coude"] = np.mean(cycles, axis=0)
    std_per_joint["flexion coude"] = np.std(cycles, axis=0)
    # ===========================
    # 3) SUBPLOTS
    # ===========================

    x = np.linspace(0, 100, 200)

    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

    for ax, (name, cycles) in zip(axes, cycles_per_joint.items()):

        # cycles individuels
        for c in cycles:
            ax.plot(x, c, color="gray", alpha=0.3)

        # moyenne
        ax.plot(x, mean_per_joint[name], color="blue", linewidth=2)

        # écart-type
        ax.fill_between(
            x,
            mean_per_joint[name] - std_per_joint[name],
            mean_per_joint[name] + std_per_joint[name],
            color="blue", alpha=0.2
        )

        ax.set_title(name)
        ax.set_ylabel("Angle (°)")
        ax.grid(True)

    axes[-1].set_xlabel("Cycle (%)")

    plt.tight_layout()
    plt.show()



    # Animate the results if biorbd viz is installed
    if show and biorbd_viz_found:
        b = bioviz.Viz(loaded_model=model)
        b.load_movement(q_recons)
        b.exec()

if __name__ == "__main__":
    main(show=True)
