import numpy as np
import time
import casadi as ca
import biorbd
import biorbd_casadi as biorbdc
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import warnings

MODE_PEDALAGE = "eccentric"
PUISSANCE = "40"

MODEL_PATH = "/Users/leo/Desktop/Projet/modele_opensim/wu_bras_gauche_seth_left_Sidonie_vtp.bioMod"

Q_PATH    = f"/Users/leo/Desktop/Projet/Collecte_25_11/{MODE_PEDALAGE}_{PUISSANCE}W/q_inverse_kinematic.npy"
QDOT_PATH = f"/Users/leo/Desktop/Projet/Collecte_25_11/{MODE_PEDALAGE}_{PUISSANCE}W/qdot_inverse_kinematic.npy"
TAU_PATH  = f"/Users/leo/Desktop/Projet/Collecte_25_11/{MODE_PEDALAGE}_{PUISSANCE}W/tau_inverse_dynamic.npy"
EMG_PATH  = f"/Users/leo/Desktop/Projet/Collecte_25_11/{MODE_PEDALAGE}_{PUISSANCE}W/emg_processed_resampled.npy"

FIRST, END = 3000, 3200
RES_BND_know = 2.5
RES_BND_unknow = 4
TAU_RES_BND = np.concatenate((
    RES_BND_unknow * np.ones(5),
    RES_BND_know   * np.ones(5)
))
EPS_ACT = 1e-6

if MODE_PEDALAGE == "concentric":
    W_TAU = 2.6e10  #concentric
    W_RES = 1.3e8
    W_EMG = 2.5e10
    W_ACT = 2.5e8
    SAT = 9

elif MODE_PEDALAGE == "eccentric":
    W_TAU = 2.6e10  # concentric
    W_RES = 1.3e8
    W_EMG = 2.5e10
    W_ACT = 2.5e8
    SAT = 9


else:
    print("PROBLEME MODE DE PEDALAGE")

DELAY = 30 # en ms (EMG en avance sur activation) : facteur 10

active_dof = [6,7,8,9,10,11,12,13,14,15]

emg_to_muscle = {
    0: "DeltoideusClavicle",
    1: "DeltoideusScapula_M",
    2: "DeltoideusScapula_P",
    3: "TrapeziusScapula_S",
    4: "TRI",
    5: "BIC",
    6: "TrapeziusScapula_M",
    7: "TrapeziusScapula_I",
    8: "LatissimusDorsi",
    9: "Pectoralis",
    #10: "brachio" #pas dans le modèle
}


def transpose_if_needed(arr, target_rows):
    return arr if arr.shape[0] == target_rows else arr.T

def ms_to_frame(delay):
    n_frame = delay/10
    return n_frame


def extract_cycles_generic(signal, peaks):
    out = []
    for i in range(len(peaks) - 1):
        seg = signal[peaks[i]:peaks[i + 1]]
        seg_norm = np.interp(
            np.linspace(0, 1, 200),
            np.linspace(0, 1, len(seg)),
            seg
        )
        out.append(seg_norm)
    return np.array(out)

def build_emg_to_muscle_mapping(model: biorbd.Model, emg_to_muscle_dict: dict, verbose=True):
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

def build_nlp_solver(model_path, nb_mus, nb_tau):
    model = biorbdc.Model(model_path)

    a = ca.MX.sym("a", nb_mus)
    tau_res = ca.MX.sym("tau_res", nb_tau)
    x = ca.vertcat(a, tau_res)

    q     = ca.MX.sym("q", model.nbQ())
    qdot  = ca.MX.sym("qdot", model.nbQ())
    tauID = ca.MX.sym("tauID", nb_tau)
    emg   = ca.MX.sym("emg", nb_mus)
    mask  = ca.MX.sym("mask", nb_mus)

    w_tau, w_res, w_emg, w_act = ca.MX.sym("w_tau"), ca.MX.sym("w_res"), ca.MX.sym("w_emg"), ca.MX.sym("w_act")

    p = ca.vertcat(q, qdot, tauID, emg, mask, w_tau, w_res, w_emg, w_act)

    states = model.stateSet()
    for i in range(model.nbMuscles()):
        states[i].setActivation(a[i])

    tau_m_full = model.muscularJointTorque(states, q, qdot).to_mx()
    tau_m = tau_m_full[active_dof]

    tau_err = tauID - (tau_m + tau_res)
    emg_err = (emg - a) * mask
    a_free  = a * (1 - mask)

    cost = (
        w_tau * ca.sumsqr(tau_err) +
        w_res * ca.sumsqr(tau_res) +
        w_emg * ca.sumsqr(emg_err) +
        w_act * ca.sumsqr(a_free) +
        ca.sum(a**SAT)
    )

    solver = ca.nlpsol(
        "solver", "ipopt",
        {"x": x, "f": cost, "p": p},
        {"ipopt.print_level": 5, "print_time": True}
    )

    f_cost = ca.Function("f_cost", [x, p], [cost])

    return solver, f_cost

def main():

    model_np = biorbd.Model(MODEL_PATH)
    nbQ   = model_np.nbQ()
    nbMus = model_np.nbMuscles()
    nbTau = len(active_dof)

    q    = np.load(Q_PATH)
    qdot = np.load(QDOT_PATH)
    tau  = np.load(TAU_PATH)
    emg  = np.load(EMG_PATH)

    if q.shape[1] == emg.shape[1]:
        print("Trigger bien détecté")
    else :
        warnings.warn("TRIGGER DIFFERENT SUR Q ET EMG : VERIFIER !!")

    q    = q[:, FIRST:END]
    qdot = qdot[:, FIRST:END]
    tau  = tau[active_dof, FIRST:END]
    emg  = emg[:, int(FIRST+ms_to_frame(DELAY)):int(END+ms_to_frame(DELAY))]

    print("q shape   :", q.shape)
    print("emg shape :", emg.shape)
    print("FIRST,END :", FIRST, END)

    track_idx, emg_idx, tracked_names = build_emg_to_muscle_mapping(model_np, emg_to_muscle)

    is_emg_mask = np.zeros(nbMus)
    is_emg_mask[track_idx] = 1.0

    solver, f_cost = build_nlp_solver(MODEL_PATH, nbMus, nbTau)

    n_frames = q.shape[1]
    mus_act  = np.zeros((nbMus, n_frames))
    tau_res  = np.zeros((nbTau, n_frames))
    tau_musc = np.zeros((nbTau, n_frames))
    tau_err = np.zeros((nbTau, n_frames))
    mus_force = np.zeros((nbMus, n_frames))


    t0 = time.time()

    for k in range(n_frames):

        emg_full = np.zeros(nbMus)
        emg_full[track_idx] = emg[emg_idx, k]

        p = np.concatenate([
            q[:,k], qdot[:,k], tau[:,k],
            emg_full, is_emg_mask,
            [W_TAU, W_RES, W_EMG, W_ACT]
        ])

        x0 = np.concatenate([0.1*np.ones(nbMus), np.zeros(nbTau)])
        lbx = np.concatenate([np.zeros(nbMus), -TAU_RES_BND])
        ubx = np.concatenate([np.ones(nbMus),  TAU_RES_BND])

        sol = solver(x0=x0, lbx=lbx, ubx=ubx, p=p)

        xopt = np.array(sol["x"]).squeeze()

        a_opt = np.clip(xopt[:nbMus], 0, 1-EPS_ACT)
        tau_res[:,k] = xopt[nbMus:]
        mus_act[:,k] = a_opt

        states = model_np.stateSet()
        for i in range(nbMus):
            states[i].setActivation(a_opt[i])

        tau_musc[:,k] = model_np.muscularJointTorque(
            states, q[:,k], qdot[:,k]
        ).to_array()[active_dof]

        tau_err[:,k] = tau[:,k]-(tau_musc[:,k]+tau_res[:,k])

        mus_force[:,k] = model_np.muscleForces(states, q[:,k], qdot[:,k]).to_array()

        print("Coût CasADi =", f_cost(x0, p).full())
        print("Coût Sortie =", f_cost(xopt, p).full())


        if (k+1) % 100 == 0:
            print(f"Frame {k+1}/{n_frames}")

    print("Done in", time.time() - t0, "s")
    np.save(f"/Users/leo/Desktop/Projet/Collecte_25_11/{MODE_PEDALAGE}_{PUISSANCE}W/muscle_activations_nonlinear.npy", mus_act)
    np.save(f"/Users/leo/Desktop/Projet/Collecte_25_11/{MODE_PEDALAGE}_{PUISSANCE}W/muscles_forces.npy", mus_force)

    err_mean = np.mean(np.abs(tau_err), axis=1)
    err_std = np.std(np.abs(tau_err), axis=1)
    res_mean = np.mean(np.abs(tau_res), axis=1)
    res_std = np.std(np.abs(tau_res), axis=1)


    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(nbTau), err_mean, yerr = err_std, capsize=5)
    plt.title("Moyenne de |tau_err| par DoF  (τ - (A a + τ_res))")
    plt.xlabel("DoF index")
    plt.ylabel("mean |tau_err|")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.show()

    max_tau = np.max(np.abs(tau), axis=1)
    rap_res_tau = (res_mean/max_tau)*100

    x = np.arange(nbTau)

    plt.figure(figsize=(10, 4))
    bars = plt.bar(x, res_mean, yerr=res_std, capsize=5)

    plt.title(f"Moyenne de |τ_res| par DoF  (borne ± {RES_BND_unknow} and {RES_BND_know} Nm)")
    plt.xlabel("DoF index")
    plt.ylabel("mean |τ_res|")
    plt.grid(True, axis="y")
    # =========================
    # Affichage du % au-dessus des barres
    # =========================
    for i, bar in enumerate(bars):
        if float(rap_res_tau[i]) < 15:
            A = 'green'
        elif 15 <= float(rap_res_tau[i]) < 20:
            A = 'y'
        else:
            A = 'red'
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + res_std[i] + 0.02 * np.max(res_mean),  # petit offset au-dessus
            f"{rap_res_tau[i]:.1f} %",
            ha='center',
            va='bottom',
            fontsize=9,
            fontweight='bold',
            color = A
        )
    plt.tight_layout()
    plt.show()

    nbTrackedMus = track_idx.shape[0]

    mus_act_emg = np.zeros((nbTrackedMus,n_frames))
    for i, idx in enumerate(track_idx):
        mus_act_emg[i, :] = mus_act[idx, :]

    all_muscle_names = [model_np.muscle(i).name().to_string() for i in range(model_np.nbMuscles())]


    plt.figure(figsize=(14, 6))
    for i in range(nbMus):
        plt.plot(mus_force[i, :], label=f"{i}: {all_muscle_names[i]}")
    plt.title("Forces musculaires")
    plt.xlabel("Frame")
    plt.ylabel("Force en N")
    plt.grid(True)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.show()


    ### PLOT DES TAU

    import plotly.graph_objects as go

    fig = go.Figure()

    dof_names = [s.to_string() for s in model_np.nameDof()]

    for i in range(nbTau):
        # tau
        fig.add_trace(
            go.Scatter(
                y=tau[i, :],
                mode="lines",
                name=f"{i}: (tau)",
                legendgroup=f"group_{i}",
                legendgrouptitle_text=f"{dof_names[i+6]}"
            )
        )

        # tau_res
        fig.add_trace(
            go.Scatter(
                y=tau_res[i, :],
                mode="lines",
                name=f"{i}: tau_res",
                legendgroup=f"group_{i}",
                line=dict(dash="dash")
            )
        )

        # tau_err
        fig.add_trace(
            go.Scatter(
                y=tau_musc[i, :],
                mode="lines",
                name=f"{i}: tau_act",
                legendgroup=f"group_{i}",
                line=dict(dash="dot")
            )
        )


    fig.update_layout(
        title="Couples articulaires : tau, tau_res, tau_err",
        xaxis_title="Frame",
        yaxis_title="Couple (N·m)",
        hovermode="x unified",
        legend=dict(
            itemclick="toggle",
            itemdoubleclick="toggleothers"
        ),
        template="plotly_white",
        height=500
    )

    fig.show()
    # Return results

    # PLOT COMPARATIF ACT VS EMG

    aemg = go.Figure()

    for i, name in enumerate(tracked_names):
        emg_ch = emg_idx[i]  # ✅ canal EMG correspondant à ce muscle
        # tau
        aemg.add_trace(
            go.Scatter(
                y=mus_act_emg[i, :],
                mode="lines",
                name=f"{i}: {name}",
                legendgroup=f"group_{i}"
            )
        )

        # tau_res
        aemg.add_trace(
            go.Scatter(
                y=emg[emg_ch,:],
                mode="lines",
                name=f"EMG ch{emg_ch}",
                legendgroup=f"group_{i}",
                line=dict(dash="dash")
            )
        )

    aemg.update_layout(
        title="Activation vs EMG",
        xaxis_title="Frame",
        yaxis_title="Activation",
        hovermode="x unified",
        legend=dict(
            itemclick="toggle",
            itemdoubleclick="toggleothers"
        ),
        template="plotly_white",
        height=500
    )

    aemg.show()

    # PLOT ACT

    act = go.Figure()

    for i, name in enumerate(all_muscle_names):
        # tau
        act.add_trace(
            go.Scatter(
                y=mus_act[i, :],
                mode="lines",
                name=f"{i}: {name}",
                legendgroup=f"group_{i}"
            )
        )

    act.update_layout(
        title="Activation",
        xaxis_title="Frame",
        yaxis_title="Activation",
        hovermode="x unified",
        legend=dict(
            itemclick="toggle",
            itemdoubleclick="toggleothers"
        ),
        template="plotly_white",
        height=500
    )

    act.show()
    #---------- Plot final -----------
    # ==========================================================
    # DÉTECTION DES CYCLES À PARTIR D’UN DOF DE RÉFÉRENCE
    # ==========================================================

    # Choix automatique d’un DOF de référence pour détecter les cycles
    ref_idx = 0 #8 si les 10 DoF

    print(f"DoF utilisé comme référence du cycle : coude flexion")

    # Signal de référence
    ref_signal = tau[ref_idx, :]

    # Détection des peaks
    peaks_sel, _ = find_peaks(ref_signal, distance=100)

    print("Nombre de cycles détectés :", len(peaks_sel) - 1)
    # ==========================================================
    # === SUBPLOT : COUPLES / FORCES PAR DOF SUR LE CYCLE ===
    # =========================================================
    MUSC_TO_PLOT = "ALL"
    # ----------- Sélection DoF à tracer ------------
    if MUSC_TO_PLOT == "ALL":
        selected_musc = all_muscle_names
    else:
        selected_musc = MUSC_TO_PLOT

    # ----------- Construction cycles τ par DoF --------
    cycles_tau = {}
    mean_tau = {}
    std_tau = {}

    for dof in selected_musc:
        idx = all_muscle_names.index(dof)
        cyc = extract_cycles_generic(mus_act[idx, :], peaks_sel)
        cycles_tau[dof] = cyc
        mean_tau[dof] = np.mean(cyc, axis=0)
        std_tau[dof] = np.std(cyc, axis=0)
    # ----------- Plot final : GRILLE DE SUBPLOTS -------------------------

    import math

    x = np.linspace(0, 100, 200)


    # Définition automatique d'une grille
    n_cols = math.ceil(math.sqrt(nbMus))
    n_rows = math.ceil(nbMus / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), sharex=True)
    axes = axes.flatten()  # pour parcourir facilement

    for ax, dof in zip(axes, all_muscle_names):

        # cycles individuels
        for c in cycles_tau[dof]:
            ax.plot(x, c, color="gray", alpha=0.25)

        # moyenne
        ax.plot(x, mean_tau[dof], linewidth=2, color="blue")

        # écart-type
        ax.fill_between(
            x,
            mean_tau[dof] - std_tau[dof],
            mean_tau[dof] + std_tau[dof],
            color="blue",
            alpha=0.15
        )

        ax.set_title(dof, fontsize=8)
        ax.set_ylabel("Activation")
        ax.grid(True)

    # Supprimer les axes inutilisés si la grille est trop grande
    for i in range(len(selected_musc), len(axes)):
        fig.delaxes(axes[i])

    # Label global
    plt.xlabel("Cycle (%)")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
