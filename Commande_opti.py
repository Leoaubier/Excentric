import numpy as np
import biorbd
from bioptim import (
    OptimalControlProgram,
    DynamicsList,
    DynamicsFcn,
    ObjectiveList,
    ObjectiveFcn,
    ConstraintList,
    ConstraintFcn,
    BoundsList,
    InitialGuessList,
    OdeSolver,
    Solver,
    ControlType,
    Node,
    ExternalForceSetTimeSeries,
)

MODE_PEDALAGE = "concentric"
PUISSANCE = "40"

# =====================================================
# 1. Chargement du modèle
# =====================================================
model_path = "/Users/leo/Desktop/Projet/modele_opensim/wu_bras_gauche_Sidonie_last.bioMod"
model = biorbd.Model(model_path)

n_q = model.nbQ()
n_mus = model.nbMuscles()

# =====================================================
# 2. Chargement des données expérimentales
# =====================================================
q = np.load(f"/Users/leo/Desktop/Projet/Collecte_25_11/{MODE_PEDALAGE}_{PUISSANCE}W/q_inverse_kinematic.npy")            # (nQ, nFrames)
qdot = np.load(f"/Users/leo/Desktop/Projet/Collecte_25_11/{MODE_PEDALAGE}_{PUISSANCE}W/qdot_inverse_kinematic.npy")      # (nQ, nFrames)

pedal_forces = np.load(f"/Users/leo/Desktop/Projet/Collecte_25_11/{MODE_PEDALAGE}_{PUISSANCE}W/constraint_global.npy")
# shape attendue : (6, nFrames)
# [Fx, Fy, Fz, Mx, My, Mz] dans le repère global

assert q.shape[1] == pedal_forces.shape[1], "Incohérence temporelle"

n_frames = q.shape[1]
n_shooting = n_frames - 1
final_time = 1.0  # normalisé (ou durée réelle du cycle)

# =====================================================
# 3. Définition des forces externes (pédale → main)
# =====================================================
external_forces = ExternalForceSetTimeSeries()

external_forces.add(
    segment_name="hand",               # DOIT correspondre au bioMod
    point_of_application="hand_pedal", # marker / node du bioMod
    force=pedal_forces,
)

# =====================================================
# 4. Définition des dynamiques
# =====================================================
dynamics = DynamicsList()
dynamics.add(
    DynamicsFcn.MUSCLE_ACTIVATIONS_DRIVEN,
    external_forces=external_forces,
)

# =====================================================
# 5. Fonction objectif
# =====================================================
objectives = ObjectiveList()

# Minimisation de l'effort musculaire
objectives.add(
    ObjectiveFcn.Lagrange.MINIMIZE_CONTROL,
    key="muscles",
    weight=1.0,
)

# (Optionnel) Tracking EMG
# emg = np.load("/mnt/data/emg_processed_resampled.npy")
# objectives.add(
#     ObjectiveFcn.Lagrange.TRACK_CONTROL,
#     key="muscles",
#     target=emg,
#     weight=10.0,
# )

# =====================================================
# 6. Contraintes cinématiques strictes
# =====================================================
constraints = ConstraintList()

constraints.add(
    ConstraintFcn.TRACK_STATE,
    key="q",
    node=Node.ALL,
    target=q,
)

constraints.add(
    ConstraintFcn.TRACK_STATE,
    key="qdot",
    node=Node.ALL,
    target=qdot,
)

# =====================================================
# 7. Bornes sur les états / contrôles
# =====================================================
bounds = BoundsList()

bounds.add(
    "muscles",
    min_bound=0.0,
    max_bound=1.0,
)

initial_guess = InitialGuessList()
initial_guess.add(
    "muscles",
    initial_guess=0.1,
)

# =====================================================
# 8. Création du problème de contrôle optimal
# =====================================================
ocp = OptimalControlProgram(
    model=model,
    dynamics=dynamics,
    n_shooting=n_shooting,
    phase_time=final_time,
    objective_functions=objectives,
    constraints=constraints,
    bounds=bounds,
    initial_guess=initial_guess,
    ode_solver=OdeSolver.RK4(),
    control_type=ControlType.CONSTANT,
)

# =====================================================
# 9. Résolution
# =====================================================
solver = Solver.IPOPT()
solver.set_maximum_iterations(1000)
solver.set_tolerance(1e-6)
solver.set_print_level(5)

solution = ocp.solve(solver)

# =====================================================
# 10. Extraction des activations musculaires
# =====================================================
activations = solution.controls["muscles"]  # (nMuscles, n_shooting)

# =====================================================
# 11. Calcul des forces musculaires
# =====================================================
muscle_forces = np.zeros((n_mus, n_shooting))

for k in range(n_shooting):
    model.updateMuscles(
        q[:, k],
        qdot[:, k],
        activations[:, k],
    )
    for m in range(n_mus):
        muscle_forces[m, k] = model.muscle(m).force(model, k)

#np.save("estimated_muscle_forces.npy", muscle_forces)
#np.save("estimated_activations.npy", activations)

print("=== Estimation terminée ===")
