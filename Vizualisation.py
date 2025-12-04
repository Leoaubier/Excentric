import biorbd
import bioviz
import numpy as np

model_path = '/Users/leo/Desktop/Projet/modele_opensim/wu_bras_gauche_seth_left_Sidonie.bioMod'
movement_path = '/Users/leo/Desktop/Projet/Collecte_25_11/IK/inverse_kinematic_sidonie_40W.npy'
force_path = '/Users/leo/Desktop/Projet/Collecte_25_11/IK/constraint_global_40W.npy'

model = biorbd.Model(str(model_path))

print(np.load(force_path)[0,:,1].shape)
force_frame = np.load(force_path)[0:1,:,:]
force_frame = np.concatenate((np.zeros((1, 3, force_frame.shape[2])), force_frame), axis=1)

b = bioviz.Viz(loaded_model=model, show_meshes=False,
    show_markers=True,
    show_muscles=False,
    show_analyses_panel=True)
b.load_movement(np.load(movement_path))
b.load_experimental_forces(force_frame, segments=["hand_left"], normalization_ratio=1)
b.exec()

