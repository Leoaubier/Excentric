from pyorerun import BiorbdModel, PhaseRerun
import numpy as np
import biorbd


ESSAI = "Collecte_18_03"
PUISSANCE = "60"

frame_con = 4028   #  frame concentrique
frame_ecc = 15036   #  frame excentrique


model_path = f"/Users/leo/Desktop/Projet/{ESSAI}/model_{ESSAI}_con.bioMod"
model_pedal_path = f'/Users/leo/Desktop/Projet/modele_opensim/model_pedal.bioMod'

q_con = np.load(f'/Users/leo/Desktop/Projet/{ESSAI}/concentric_{PUISSANCE}W/q_inverse_kinematic.npy')
q_ecc = np.load(f'/Users/leo/Desktop/Projet/{ESSAI}/eccentric_{PUISSANCE}W/q_inverse_kinematic.npy')

q_pedal_con = np.load(f'/Users/leo/Desktop/Projet/{ESSAI}/concentric_{PUISSANCE}W/inverse_kinematic_pedal.npy')
q_pedal_ecc = np.load(f'/Users/leo/Desktop/Projet/{ESSAI}/eccentric_{PUISSANCE}W/inverse_kinematic_pedal.npy')


model_con = BiorbdModel(model_path)
model_ecc = BiorbdModel(model_path)

pedal_con = BiorbdModel(model_pedal_path)
pedal_ecc = BiorbdModel(model_pedal_path)


q_con_frame = q_con[:, frame_con][:, None]
q_ecc_frame = q_ecc[:, frame_ecc][:, None]

q_pedal_con_frame = q_pedal_con[:, frame_con][:, None]
q_pedal_ecc_frame = q_pedal_ecc[:, frame_ecc][:, None]

t = np.array([0])  # une seule frame

viz = PhaseRerun(t)


viz.add_animated_model(
    model_con,
    q_con_frame,
)

viz.add_animated_model(
    model_ecc,
    q_ecc_frame,
)

# pédales
viz.add_animated_model(
    pedal_con,
    q_pedal_con_frame,
)

viz.add_animated_model(
    pedal_ecc,
    q_pedal_ecc_frame,
)



viz.rerun("Comparaison frame CON vs ECC")