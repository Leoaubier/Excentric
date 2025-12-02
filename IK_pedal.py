from pathlib import Path
from math import pi
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
START_FRAME = 3500       # Première frame incluse
END_FRAME   = 8000    # Dernière frame (None = dernière frame du fichier)


# === 1. Markers du modèle, DANS L'ORDRE DU .bioMod ===
MODEL_MARKERS = [
    "Crank_Axe_L",
    "Crank_Axe_R",
    "Pedal4",
    "Pedal5",
    "Pedal6",
]

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

    model_path = Path("/Users/leo/Desktop/Projet/modele_opensim/model_pedal.bioMod")
    c3d_path = Path("/Users/leo/Desktop/Projet/Collecte_25_11/C3D_labelled/concentric_40W.c3d")

    model = biorbd.Model(str(model_path))
    nq = model.nbQ()
    print(nq)

    c3d = ezc3d.c3d(str(c3d_path))
    raw_markers = c3d["data"]["points"][:3, :, :]
    labels = c3d["parameters"]["POINT"]["LABELS"]["value"]
    units = c3d["parameters"]["POINT"]["UNITS"]["value"][0]

    if units.lower().startswith("mm"):
        raw_markers /= 1000.0

    mapping = build_marker_mapping(labels)
    markers = extract_relevant_markers(raw_markers, mapping)

    # === APPLY FRAME SELECTION HERE ===
    markers = markers[:, :, START_FRAME:END_FRAME]
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


    # === Extraction coude ===
    # ===========================
    # CHOIX DES DOF À ANALYSER
    # ===========================

    JOINTS = {
        "crank_rot": 0,
        "pedal_rot": 1,
    }

    # ===========================
    # 1) plot sequence
    # ===========================

    plt.plot(np.rad2deg(q_recons[3, :]), label="crank_rot")
    plt.plot(np.rad2deg(q_recons[4, :]), label="pedal_rot")

    # ===========================
    # 2) Enregistrement des données
    # ===========================
    np.save("/Users/leo/Desktop/Projet/Collecte_25_11/IK/inverse_kinematic_pedal_40W.npy", q_recons)
    print("données IK enregistrées :)")




    q_recons_r = q_recons.copy()

    # Corrige toutes les articulations (si besoin)
    q_recons_r[3, :] = np.unwrap(q_recons_r[3, :])
    q_recons_r[4, :] = np.unwrap(q_recons_r[4, :])

    plt.plot(np.rad2deg(q_recons_r[3, :]), label="crank_rot")
    plt.plot(np.rad2deg(q_recons_r[4, :]), label="pedal_rot")
    plt.legend()
    plt.show()

    # Animate the results if biorbd viz is installed
    if show and biorbd_viz_found:
        b = bioviz.Viz(loaded_model=model)
        b.load_movement(q_recons_r)
        b.exec()

if __name__ == "__main__":
    main(show=True)
