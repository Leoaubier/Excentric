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

    biorbd_viz_found = True
except ModuleNotFoundError:
    biorbd_viz_found = False


# === Choix des frames à analyser ===
END_FRAME   = 8000    # Dernière frame (None = dernière frame du fichier)


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


def numpy_markers_to_nodes(markers_frame):
    nodes = []
    for i in range(markers_frame.shape[1]):
        x, y, z = markers_frame[:, i]
        node = biorbd.NodeSegment(float(x), float(y), float(z))
        nodes.append(node)
    return nodes


def to_dic(all_data_int):
    dic_data = {"time": all_data_int[0, :],
                "LFX": all_data_int[1, :],
                "LFY": all_data_int[2, :],
                "LFZ": all_data_int[3, :],
                "LMX": all_data_int[4, :],
                "LMY": all_data_int[5, :],
                "LMZ": all_data_int[6, :],
                "LAX": all_data_int[7, :],
                "LAY": all_data_int[8, :],
                "left_pedal_angle": all_data_int[17, :],
                "crank_angle": all_data_int[19,:]
                }
    return dic_data

def transform_forces_to_global(model, q_recons, F_local, M_local, ratio=2): #ratio entre la frequence d'echantillonage
    # des pédales 200Hz et les frames 100Hz
    n_frames = q_recons.shape[1]

    F_global = np.zeros((3, n_frames))
    M_global = np.zeros((3, n_frames))

    for i in range(n_frames):

        # Prendre la bonne valeur de force : 2000 Hz (trigger) → 100 Hz (frames)
        Fp = F_local[:, i * ratio]
        Mp = M_local[:, i * ratio]

        # Coordonnées généralisées
        q_i = biorbd.GeneralizedCoordinates(q_recons[:, i])

        # Roto-transformation globale du segment
        T = model.globalJCS(q_i, "Pedal_left").to_array()
        R = T[:3, :3]
        p = T[:3, 3]

        # Transformation de la force
        Fg = R.T @ Fp
        # Transformation du moment
        Mg = R.T @ Mp + np.cross(p, Fg)

        F_global[:, i] = Fg
        M_global[:, i] = Mg

    return F_global, M_global



def main(show=True):

    model_path = Path("/Users/leo/Desktop/Projet/modele_opensim/model_pedal.bioMod")
    c3d_path = Path("/Users/leo/Desktop/Projet/Collecte_25_11/C3D_labelled/concentric_40W.c3d")
    sensix_path = Path("/Users/leo/Desktop/Projet/Collecte_25_11/pedales/Results-concentric_40w_001.lvm")


    #afficher_entetes_ezc3d(str(c3d_path))

    model = biorbd.Model(str(model_path))
    nq = model.nbQ()
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
    markers = markers[:, :, find_trigger(str(c3d_path)):END_FRAME]
    n_frames = markers.shape[2]
    kalman = biorbd.KalmanReconsMarkers(model)

    q_recons = np.zeros((nq, n_frames))

    for i in range(n_frames):
        marker_nodes = numpy_markers_to_nodes(markers[:, :, i])

        q = biorbd.GeneralizedCoordinates(model)
        qdot = biorbd.GeneralizedVelocity(model)
        qddot = biorbd.GeneralizedAcceleration(model)

        kalman.reconstructFrame(model, marker_nodes, q, qdot, qddot)
        q_recons[:, i] = q.to_array()

        if i % 200 == 0:
            print(f"Frame {i}/{n_frames}")



    print("IK terminé.")

    np.save("/Users/leo/Desktop/Projet/Collecte_25_11/IK/inverse_kinematic_pedal_40W.npy", q_recons)
    print("données IK enregistrées :)")

    all_data = []
    with open(sensix_path, 'r') as f:
        csvreader = csv.reader(f, delimiter='\n')
        for row in csvreader:
            all_data.append(np.array(row[0].split("\t")))
    all_data = np.array(all_data, dtype=float).T

    global_force, global_moment = transform_forces_to_global(model, q_recons, all_data[1:4,:], all_data[4:7,:])
    global_constraint = [global_moment, global_force]
    np.save("/Users/leo/Desktop/Projet/Collecte_25_11/IK/constraint_global_40W.npy", global_constraint)
    print("Forces et Moments enregistrés")
    print("markers frames:", markers.shape[2])
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
        b = bioviz.Viz(loaded_model=model)
        b.load_movement(q_recons)
        b.exec()

if __name__ == "__main__":
    main(show=True)
