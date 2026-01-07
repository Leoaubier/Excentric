import biorbd_casadi as biorbd
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# PARAMÈTRES
# =========================================================
CALCUL = True

frame_start = 4000
frame_end   = 4100

DOF_START = 6
DOF_END   = 16   # python slice

w_emg = 1e14
w_a   = 10.0
w_tau = 100.0
w_res = 10.0

# =========================================================
# OUTILS
# =========================================================
def get_R_and_Fiso(model, q_frame):
    nb_mus = model.nbMuscles()

    # Convertir le Jacobien en CasADi MX
    J = model.musclesLengthJacobian(q_frame).to_mx()  # (nbMus, nbQ)
    R = -ca.transpose(J)                               # Transposée avec CasADi

    # Force isométrique maximale
    Fiso = ca.vertcat(*[model.muscle(i).characteristics().forceIsoMax() for i in range(nb_mus)])

    return R, Fiso



def build_emg_to_muscle_mapping(model, emg_to_muscle):
    muscle_names = [model.muscle(i).name().to_string()
                    for i in range(model.nbMuscles())]

    track_idx, emg_src_idx, tracked_names = [], [], []

    for emg_ch, key in sorted(emg_to_muscle.items()):
        for i, name in enumerate(muscle_names):
            if key in name:
                track_idx.append(i)
                emg_src_idx.append(emg_ch)
                tracked_names.append(name)

    return np.array(track_idx), np.array(emg_src_idx), tracked_names

# =========================================================
# PATHS
# =========================================================
MODEL_PATH = "/Users/leo/Desktop/Projet/modele_opensim/wu_bras_gauche_seth_left_Sidonie.bioMod"
Q_PATH     = "/Users/leo/Desktop/Projet/Collecte_25_11/IK/q_inverse_kinematic_sidonie_40W.npy"
QDOT_PATH  = "/Users/leo/Desktop/Projet/Collecte_25_11/IK/qdot_inverse_kinematic_sidonie_40W.npy"
TAU_PATH   = "/Users/leo/Desktop/Projet/Collecte_25_11/ID/tau_inverse_dynamic_Sidonie_40w.npy"
EMG_PATH   = "/Users/leo/Desktop/Projet/Collecte_25_11/EMG/emg_processed_resampled.npy"

# =========================================================
# CHARGEMENT
# =========================================================
model = biorbd.Model(MODEL_PATH)

q    = np.load(Q_PATH)
qdot = np.load(QDOT_PATH)
tau  = np.load(TAU_PATH)
emg  = np.load(EMG_PATH)

n_frames   = frame_end - frame_start
n_muscles  = model.nbMuscles()
n_q        = model.nbQ()
n_dof      = DOF_END - DOF_START

# =========================================================
# EMG → MUSCLES
# =========================================================
emg_to_muscle = {
    0: "DeltoideusClavicle",
    1: "DeltoideusScapula_M",
    2: "DeltoideusScapula_P",
    3: "TRI_",
    4: "BIC_",
    5: "TrapeziusScapula_M",
    6: "TrapeziusScapula_S",
    7: "TrapeziusScapula_I",
    8: "LatissimusDorsi",
    9: "PectoralisMajor",
}

track_idx, emg_src_idx, tracked_names = build_emg_to_muscle_mapping(
    model, emg_to_muscle
)

# =========================================================
# SYMBOLIQUE CASADI
# =========================================================
a        = ca.MX.sym("a", n_muscles)
tau_res  = ca.MX.sym("tau_res", n_dof)

q_sym    = ca.MX.sym("q", n_q)
tau_sym  = ca.MX.sym("tau", n_dof)
emg_sym  = ca.MX.sym("emg", len(emg_to_muscle))
Fiso_sym = ca.MX.sym("Fiso", n_muscles)
R_sym    = ca.MX.sym("R", n_dof, n_muscles)

# forces musculaires
f_m = a * Fiso_sym

# couples musculaires
tau_m = R_sym @ f_m

# =========================================================
# COÛT
# =========================================================
cost_emg = 0
for emg_ch in emg_to_muscle:
    mus = [i for i, src in zip(track_idx, emg_src_idx) if src == emg_ch]
    if mus:
        cost_emg += ca.sumsqr(ca.vertcat(*[a[m] for m in mus]) - emg_sym[emg_ch])

cost = (
    w_emg * cost_emg
    + w_a   * ca.sumsqr(a)
    + w_tau * ca.sumsqr(tau_sym - (tau_m + tau_res))
    + w_res * ca.sumsqr(tau_res)
)

# =========================================================
# NLP
# =========================================================
x = ca.vertcat(a, tau_res)
p = ca.vertcat(emg_sym, q_sym, tau_sym, Fiso_sym, ca.reshape(R_sym, -1, 1))

solver = ca.nlpsol(
    "solver", "ipopt",
    {"x": x, "f": cost, "p": p},
    {"ipopt.print_level": 0}
)

# =========================================================
# RÉSOLUTION
# =========================================================
a_sol = np.zeros((n_muscles, n_frames))
x0 = np.zeros(n_muscles + n_dof)

for k, frame in enumerate(range(frame_start, frame_end)):

    R_full, Fiso = get_R_and_Fiso(model, q[:, frame])
    R = R_full[DOF_START:DOF_END, :]

    tau_frame = tau[DOF_START:DOF_END, frame]
    tau_max = R @ Fiso
    tau_max[np.abs(tau_max) < 1e-6] = 1.0
    tau_norm = tau_frame / tau_max

    p_k = np.concatenate((
        emg[:, frame][list(emg_to_muscle.keys())],
        q[:, frame],
        tau_norm,
        Fiso,
        R.flatten()
    ))

    sol = solver(x0=x0, p=p_k)
    x_opt = sol["x"].full().squeeze()

    a_sol[:, k] = x_opt[:n_muscles]
    x0 = x_opt

# =========================================================
# PLOTS
# =========================================================
plt.figure(figsize=(12,6))
plt.plot(a_sol.T)
plt.title("Activations musculaires")
plt.grid()
plt.show()
