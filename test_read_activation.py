import re
import numpy as np
import matplotlib.pyplot as plt
import biorbd



def read_activation_file(filepath):
    """
    Lecture robuste d'un fichier RTF contenant :
    Frame i: [v1 v2 v3 ...]

    Returns
    -------
    data : np.ndarray (n_frames, n_muscles)
    """
    with open(filepath, "r", errors="ignore") as f:
        text = f.read()

    # Extraire le contenu entre crochets pour chaque frame
    frames = re.findall(r"Frame\s+\d+\s*:\s*\[(.*?)\]", text, re.DOTALL)

    data_list = []

    for frame in frames:
        # Extraire UNIQUEMENT les nombres (float, scientific notation incluse)
        numbers = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", frame)
        if numbers:
            data_list.append(np.array(numbers, dtype=float))

    if not data_list:
        raise ValueError("Aucune frame valide trouvée dans le fichier.")

    # Harmonisation de la longueur
    min_len = min(len(v) for v in data_list)
    data = np.array([v[:min_len] for v in data_list])

    return data


def main():

    model = biorbd.Biorbd(f"/Users/leo/Desktop/Projet/modele_opensim/wu_bras_gauche_seth_left_Sidonie.bioMod")
    muscles = [model.muscles[i].name
               for i in range(model.muscles.__len__())]
    path = "/Users/leo/Desktop/Projet/Collecte_25_11/statique/actcivation.rtf"
    q = np.load("/Users/leo/Desktop/Projet/Collecte_25_11/IK/q_inverse_kinematic_sidonie_40W.npy")
    qdot = np.load("/Users/leo/Desktop/Projet/Collecte_25_11/IK/qdot_inverse_kinematic_sidonie_40W.npy")
    qddot = np.load("/Users/leo/Desktop/Projet/Collecte_25_11/IK/qddot_inverse_kinematic_sidonie_40W.npy")
    tau = np.load("/Users/leo/Desktop/Projet/Collecte_25_11/ID/tau_inverse_dynamic_Sidonie_40w.npy")

    a = read_activation_file(path)
    print(a.shape)

    fig, axes = plt.subplots(7, 5, figsize=(4 * 7, 3 * 5), sharex=True)
    axes = axes.flatten()  # pour parcourir facilement

    for ax, (i, dof) in zip(axes, enumerate(muscles)):
        ax.plot(a[:, i])
        ax.set_title(dof, fontsize=8)
        ax.set_ylabel("V", fontsize=5)
        ax.grid(True)
    plt.show()

    START = 2000
    END = 3000 #--> config avec opti static
    force = np.zeros((model.muscles.__len__(), 1000))
    for i in range(START, END):
        model.muscles.update_geometry(q=q[:, i], qdot=qdot[:, i])
        # From that point on, it is possible (and recommended) to avoid sending any q or qdot to any muscle function.
        # Otherwise, the geometry will be updated again, which is not necessary and can be computationally intensive

        # Set the activations of the muscles
        model.muscles.activations = a[i-START, :]

        # We can now compute the muscles forces
        force[:,i-START] = model.muscles.forces()
    print(f"Muscle forces: {force}")

    fig, axes = plt.subplots(7, 5, figsize=(4 * 7, 3 * 5), sharex=True)
    axes = axes.flatten()  # pour parcourir facilement

    for ax, (i, dof) in zip(axes, enumerate(muscles)):
        ax.plot(force[i, :])
        ax.set_title(dof, fontsize=8)
        ax.set_ylabel("N", fontsize=5)
        ax.grid(True)
    plt.show()





if __name__ == "__main__":
    main()