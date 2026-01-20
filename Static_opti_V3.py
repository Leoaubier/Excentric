import numpy as np
import time
import casadi as ca
import biorbd
import biorbd_casadi as biorbdc
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

MODEL_PATH = "/Users/leo/Desktop/Projet/modele_opensim/wu_bras_gauche_seth_left_Sidonie.bioMod"

Q_PATH    = "/Users/leo/Desktop/Projet/Collecte_25_11/IK/q_inverse_kinematic_sidonie_40W.npy"
QDOT_PATH = "/Users/leo/Desktop/Projet/Collecte_25_11/IK/qdot_inverse_kinematic_sidonie_40W.npy"
TAU_PATH  = "/Users/leo/Desktop/Projet/Collecte_25_11/ID/tau_inverse_dynamic_Sidonie_40w.npy"
EMG_PATH  = "/Users/leo/Desktop/Projet/Collecte_25_11/EMG/emg_processed_resampled.npy"

FIRST, END = 3000, 3050
TAU_RES_BND = 3.0
EPS_ACT = 1e-6

W_TAU = 1e1
W_RES = 1e1
W_EMG = 1e10
W_ACT = 1e1

active_dof = [6,7,8,9,10,11,12,13,14,15]

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

def transpose_if_needed(arr, target_rows):
    return arr if arr.shape[0] == target_rows else arr.T

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
        w_act * ca.sumsqr(a_free)
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

    q    = q[:, FIRST:END]
    qdot = qdot[:, FIRST:END]
    tau  = tau[active_dof, FIRST:END]
    emg  = emg[:, FIRST:END]

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
        lbx = np.concatenate([np.zeros(nbMus), -TAU_RES_BND*np.ones(nbTau)])
        ubx = np.concatenate([np.ones(nbMus),  TAU_RES_BND*np.ones(nbTau)])


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
    np.save("muscle_activations_nonlinear.npy", mus_act)

    err_mean = np.mean(np.abs(tau_err), axis=1)
    res_mean = np.mean(np.abs(tau_res), axis=1)

    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(nbTau), err_mean)
    plt.title("Moyenne de |tau_err| par DoF  (τ - (A a + τ_res))")
    plt.xlabel("DoF index")
    plt.ylabel("mean |tau_err|")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.bar(np.arange(nbTau), res_mean)
    plt.title("Moyenne de |τ_res| par DoF  (borne ±5 Nm)")
    plt.xlabel("DoF index")
    plt.ylabel("mean |τ_res|")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.show()

    nbTrackedMus = track_idx.shape[0]

    mus_act_emg = np.zeros((nbTrackedMus,n_frames))
    for i, idx in enumerate(track_idx):
        mus_act_emg[i, :] = mus_act[idx, :]
    plt.figure(figsize=(14, 6))
    for i in range(nbTrackedMus):
        plt.plot(mus_act_emg[i, :], label=f"{i}: {tracked_names[i]}")
    plt.title("Activations musculaires")
    plt.xlabel("Frame")
    plt.ylabel("Activation")
    plt.grid(True)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.show()


    all_muscle_names = [model_np.muscle(i).name().to_string() for i in range(model_np.nbMuscles())]


    plt.figure(figsize=(14, 6))
    for i in range(nbMus):
        plt.plot(mus_act[i, :], label=f"{i}: {all_muscle_names[i]}")
    plt.title("Activations musculaires")
    plt.xlabel("Frame")
    plt.ylabel("Activation")
    plt.grid(True)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.show()

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

    plt.figure(figsize=(14, 6))

    for i, name in enumerate(tracked_names):
        emg_ch = emg_idx[i]  # ✅ canal EMG correspondant à ce muscle

        plt.plot(
            mus_act_emg[i, :],
            label=f"Activation {name}"
        )

        plt.plot(
            emg[emg_ch,:],
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

    import plotly.graph_objects as go

    fig = go.Figure()

    for i in range(nbTau):
        # tau
        fig.add_trace(
            go.Scatter(
                y=tau[i, :],
                mode="lines",
                name=f"{i}: (tau)",
                legendgroup=f"group_{i}"
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
