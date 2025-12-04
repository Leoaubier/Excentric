from pyorerun import BiorbdModel, PhaseRerun
import numpy as np
import biorbd

model_path = '/Users/leo/Desktop/Projet/modele_opensim/wu_bras_gauche_seth_left_Sidonie.bioMod'
model_pedal_path = '/Users/leo/Desktop/Projet/modele_opensim/model_pedal.bioMod'
movement_path = '/Users/leo/Desktop/Projet/Collecte_25_11/IK/inverse_kinematic_sidonie_40W.npy'
movement_pedal_path = '/Users/leo/Desktop/Projet/Collecte_25_11/IK/inverse_kinematic_pedal_40W.npy'
force_path = '/Users/leo/Desktop/Projet/Collecte_25_11/IK/constraint_global_40W.npy'


q = np.load(movement_path)
q_pedal = np.load(movement_pedal_path)# shape = (nQ, nFrames)
n_frames = q.shape[1]

rate = 100                               # Hz
duration = n_frames / rate               # secondes
t_span = np.linspace(0, duration, n_frames)

model = BiorbdModel(model_path)
model_pedal = BiorbdModel(model_pedal_path)
mod_ped = biorbd.Model(model_pedal_path)

force = np.load(force_path)[0,:,0:n_frames]
origin = np.zeros((3, force.shape[1]))

for i in range(n_frames):
    Qi = biorbd.GeneralizedCoordinates(q_pedal[:, i])

    jcs = mod_ped.globalJCS(Qi, "Pedal_left")
    origin[:, i] = jcs.trans().to_array()

data = np.load(force_path)               # shape (2, 3, n_frames)


viz = PhaseRerun(t_span)
viz.add_animated_model(model, q)
viz.add_animated_model(model_pedal,q_pedal)
viz.add_force_data(num=0, force_origin=origin, force_vector=force)
viz.rerun("Sidonie 40W")