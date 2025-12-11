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
END_FRAME   = None    # Dernière frame (None = dernière frame du fichier)


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

def wrap_to_180(angle_deg):
    return (np.unwrap(angle_deg) + 180) % 360 - 180

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

def main(show=True):

    model_path = Path("/Users/leo/Desktop/Projet/modele_opensim/wu_bras_gauche_seth_left_Sidonie.bioMod")
    c3d_path = Path("/Users/leo/Desktop/Projet/Collecte_25_11/C3D_labelled/concentric_40W.c3d")

    model = biorbd.Biorbd(str(model_path))
    nq = model.nb_q

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
    qdot_recons = np.zeros((nq, n_frames))
    qddot_recons = np.zeros((nq, n_frames))

    kalman = biorbd.ExtendedKalmanFilterMarkers(model, frequency=100)
    q_i, _, _ = kalman.reconstruct_frame(markers[4000])
    for i, (q_i, _, _) in enumerate(kalman.reconstruct_frames(markers)):
        q_recons[:, i] = q_i
        #qdot_recons[:, i] = qdot_i
        #qddot_recons[:, i] = qddot_i

        if i % 200 == 0:
            print(f"Frame {i}/{n_frames}")

    print("IK terminé.")


    JOINTS = {
        "Plan élévation hum": 0,
        "élévation hum": 1,
        "Rot axiale hum": 2
    }

    # ===========================
    # 1) Détection des pics via le coude (référence du cycle)
    # ===========================
    shoulder_euler = np.rad2deg(np.unwrap(q_recons[11:14, :]))
    elbow_euler = np.rad2deg(np.unwrap(q_recons[14, :]))


    plt.plot((np.rad2deg(np.unwrap(q_recons[11, :]))), label="Plan élévation hum kalmann")  #--> Abduction épaule
    plt.plot((np.rad2deg(np.unwrap(q_recons[12, :]))), label="élévation hum kalmann")  #--> Flexion épaule
    plt.plot((np.rad2deg(np.unwrap(q_recons[13, :]))), label="Rot axiale hum kalmann")  #-->
    plt.plot(elbow_euler, label="Coude kalmann")  #--> Flexion coude
    plt.legend()
    plt.show()
    # ===========================
    # 2) Enregistrement des données
    # ===========================
    np.save("/Users/leo/Desktop/Projet/Collecte_25_11/IK/q_inverse_kinematic_sidonie_40W.npy", q_recons)
    np.save("/Users/leo/Desktop/Projet/Collecte_25_11/IK/qdot_inverse_kinematic_sidonie_40W.npy", qdot_recons)
    np.save("/Users/leo/Desktop/Projet/Collecte_25_11/IK/qddot_inverse_kinematic_sidonie_40W.npy", qddot_recons)
    print("données IK enregistrées :)")

    # ===========================
    # 4) Extraction & normalisation des cycles
    # ===========================


    FIRST_FRAME_PLOT = 2000
    END_FRAME_PLOT = 6000


    peaks, _ = find_peaks(elbow_euler[FIRST_FRAME_PLOT:END_FRAME_PLOT], distance=100)

    print("Nombre de cycles détectés :", len(peaks) - 1)
    # Extraire les cycles pour chaque articulation
    cycles_per_joint = {}
    mean_per_joint = {}
    std_per_joint = {}

    for name, dof in JOINTS.items():
        #sig = np.degrees(q_recons[dof, :])
        cycles = extract_cycles(shoulder_euler[dof,FIRST_FRAME_PLOT:END_FRAME_PLOT], peaks)
        cycles_per_joint[name] = cycles
        mean_per_joint[name] = np.mean(cycles, axis=0)
        std_per_joint[name] = np.std(cycles, axis=0)

    cycles = extract_cycles(elbow_euler[FIRST_FRAME_PLOT:END_FRAME_PLOT], peaks)

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
        modelviz= biorbd.Model(str(model_path))
        b = bioviz.Viz(loaded_model=modelviz, show_local_ref_frame=True)
        b.load_movement(q_recons)
        b.exec()

if __name__ == "__main__":
    main(show=True)
