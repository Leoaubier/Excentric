import bioviz
import biorbd
import numpy as np


model_path = "/Users/leo/Desktop/Projet/modele_opensim/wu_bras_gauche_Sidonie_last.bioMod"
q_recons = np.load("/Users/leo/Desktop/Projet/Collecte_25_11/IK/q_inverse_kinematic_sidonie_40W.npy")

modelviz = biorbd.Model(str(model_path))
b = bioviz.Viz(loaded_model=modelviz)
b.exec()