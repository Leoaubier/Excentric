import biorbd_casadi as biorbd
import casadi as ca
import numpy as np
import matplotlib.pyplot as plt



CALCUL = True

def build_emg_to_muscle_mapping(model: biorbd.Model, emg_to_muscle_dict: dict, verbose=False):
    muscle_names = [model.muscle(i).name().to_string() for i in range(model.nbMuscles())]
    sorted_items = sorted(emg_to_muscle_dict.items(), key=lambda x: x[0])

    track_idx = []
    emg_src_idx = []
    tracked_muscle_names = []

    if verbose:
        print("\n[EMG → Muscle mapping]")
        print("-" * 60)

    for emg_ch, key in sorted_items:
        matches = [i for i, mname in enumerate(muscle_names) if key in mname]
        if len(matches) == 0:
            raise ValueError(f"No muscle matched for EMG key '{key}' (ch {emg_ch})")

        if verbose:
            print(f"EMG {emg_ch:>2} ({key:25s}) → {[muscle_names[i] for i in matches]}")

        for mi in matches:
            track_idx.append(mi)
            emg_src_idx.append(emg_ch)
            tracked_muscle_names.append(muscle_names[mi])

    return np.array(track_idx, dtype=int), np.array(emg_src_idx, dtype=int), tracked_muscle_names

# PATH
MODEL_PATH = "/Users/leo/Desktop/Projet/modele_opensim/wu_bras_gauche_seth_left_Sidonie.bioMod"

Q_PATH = "/Users/leo/Desktop/Projet/Collecte_25_11/IK/q_inverse_kinematic_sidonie_40W.npy"
QDOT_PATH = "/Users/leo/Desktop/Projet/Collecte_25_11/IK/qdot_inverse_kinematic_sidonie_40W.npy"
TAU_PATH = "/Users/leo/Desktop/Projet/Collecte_25_11/ID/tau_inverse_dynamic_Sidonie_40w.npy"
EMG_PATH = "/Users/leo/Desktop/Projet/Collecte_25_11/EMG/emg_processed_resampled.npy"

frame_start = 4000     # frame de début
frame_end = 4100       # frame de fin (non incluse)

w_emg = 200000.0
w_a = 25e-3
w_tau = 10.0
w_res = 1.0

n_frames = frame_end - frame_start

model = biorbd.Model(MODEL_PATH)

emg = np.load(EMG_PATH)
q = np.load(Q_PATH)
qdot = np.load(QDOT_PATH)
tau = np.load(TAU_PATH)

print("shape emg", emg.shape,"shape q", q.shape,"shape qdot", qdot.shape, "shape tau", tau.shape)

n_muscles = model.nbMuscles()
n_q = model.nbQ()
n_joints = model.nbGeneralizedTorque()

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


a = ca.MX.sym("a", n_muscles)
tau_res = ca.MX.sym("tau_res", n_joints)

q_sym = ca.MX.sym("q", n_q)
qdot_sym = ca.MX.sym("qdot", n_q)
tau_sym = ca.MX.sym("tau", n_joints)
emg_sym = ca.MX.sym("emg", len(emg_to_muscle))

print(len(emg_to_muscle))


# Forces musculaires (CasADi-compatible)
muscle_states = model.stateSet()

for i in range(n_muscles):
    muscle_states[i].setActivation(a[i])

muscle_forces = model.muscleForces(
    muscle_states,
    q_sym,
    qdot_sym
)

# Jacobien des longueurs musculaires
length_jacobian = model.musclesLengthJacobian(q_sym)  # (n_muscles, n_q)

# Couples musculaires
tau_m = ca.MX.zeros(n_joints)

for j in range(n_joints):
    for i in range(n_muscles):
        tau_m[j] += -length_jacobian.to_mx()[i, j] * muscle_forces.to_mx()[i]

cost_emg = 0
track_idx, emg_src_idx, tracked_names = build_emg_to_muscle_mapping(model, emg_to_muscle, verbose=True)

if CALCUL == True:

    # Muscles qui ont un EMG
    muscles_with_emg = set(track_idx)

    for emg_sym_idx, emg_ch in enumerate(emg_to_muscle.keys()):
        # muscles du modèle correspondant à cet EMG
        muscles_for_this_emg = [i for i, src in zip(track_idx, emg_src_idx) if src == emg_ch]
        if len(muscles_for_this_emg) == 0:
            continue  # aucun muscle pour cet EMG, on ignore

        weight = 1.0 / len(muscles_for_this_emg)
        muscle_vals = [weight * a[m] for m in muscles_for_this_emg]

        # Minimiser la différence activation - EMG
        cost_emg += ca.sumsqr(ca.vertcat(*muscle_vals) - emg_sym[emg_sym_idx])

    # Muscles sans EMG → minimiser juste l'activation²
    muscles_without_emg = [i for i in range(n_muscles) if i not in muscles_with_emg]
    cost_a = ca.sumsqr(a[muscles_without_emg]) if muscles_without_emg else 0

    # Couples musculaires
    cost_tau = ca.sumsqr(tau_sym - (tau_m + tau_res))

    # Régularisation sur tau_res
    cost_res = ca.sumsqr(tau_res)

    # Coût total
    cost = w_emg * cost_emg + w_a * cost_a + w_tau * cost_tau + w_res * cost_res

    cost_a = ca.sumsqr(a)

    cost_tau = ca.sumsqr(tau_sym - (tau_m + tau_res))

    cost_res = ca.sumsqr(tau_res)

    cost = (
        w_emg * cost_emg
        + w_a * cost_a
        + w_tau * cost_tau
        + w_res * cost_res
    )

    x = ca.vertcat(a, tau_res)

    lbx = [0.0]*n_muscles + [-5.0]*n_joints
    ubx = [1.0]*n_muscles + [5.0]*n_joints

    p = ca.vertcat(emg_sym, q_sym, qdot_sym, tau_sym)

    nlp = {
        "x": x,
        "f": cost,
        "p": p
    }

    solver = ca.nlpsol("solver", "ipopt", nlp)


    a_solution = np.zeros((n_muscles, n_frames))
    tau_res_solution = np.zeros((n_joints, n_frames))

    x0 = np.zeros(n_muscles + n_joints)


    for k, frame in enumerate(range(frame_start, frame_end)):

        p_k = np.concatenate((
            emg[:, frame][list(emg_to_muscle.keys())],  # <-- exactement 10 valeurs
            q[:, frame],
            qdot[:, frame],
            tau[:, frame]
        ))

        # --- résoudre le problème NLP ---
        sol = solver(
            x0=x0,
            lbx=lbx,
            ubx=ubx,
            p=p_k
        )

        x_opt = sol["x"].full().squeeze()

        # sauvegarder les solutions
        a_solution[:, k] = x_opt[:n_muscles]
        tau_res_solution[:, k] = x_opt[n_muscles:]


        # warm start
        x0 = x_opt

    np.save("/Users/leo/Desktop/Projet/Collecte_25_11/activations/activations_1.npy", a_solution)

else:
    a_solution = np.load("/Users/leo/Desktop/Projet/Collecte_25_11/activations/activations_1.npy")

j=0
a_emg= np.zeros((len(track_idx),n_frames))
for i in track_idx:
    a_emg[j,:] = a_solution[i,:]
    j=j+1


plt.figure(figsize=(14, 6))
for i in range(n_muscles):
    plt.plot(a_solution[i,:])
plt.title("Activations musculaires (Ceglia EMG-informed SO)")
plt.xlabel("Frame")
plt.ylabel("Activation")
plt.grid(True)
plt.legend(ncol=2, fontsize=8)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14,6))

for i, name in enumerate(tracked_names):
    emg_ch = emg_src_idx[i]  # ✅ canal EMG correspondant à ce muscle

    plt.plot(
        a_emg[i, :],
        label=f"Activation {name}"
    )

    plt.plot(
        emg[emg_ch, frame_start:frame_end],
        '--',
        label=f"EMG ch{emg_ch}"
    )

plt.xlabel("Frame")
plt.ylabel("Activation / EMG")
plt.title("Activations musculaires vs EMG")
plt.grid(True)
plt.legend(ncol=2, fontsize=8)
plt.tight_layout()
plt.show()

