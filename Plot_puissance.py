import numpy as np
import biorbd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

ESSAI = "Collecte_18_03"
PUISSANCES = ["40", "60"]
MODE = "concentric"  # ou "eccentric"


def ensure_forward_rotation(crank_angle, *signals):

    crank_angle = np.asarray(crank_angle, float)

    if np.median(np.diff(crank_angle)) < 0:
        crank_angle = crank_angle[::-1]
        signals = [s[..., ::-1] for s in signals]
        print("ECC inversé → remis dans le sens croissant")

    return (crank_angle, *signals)


def detect_cycles_from_crank(crank_angle, min_cycle_frames=30):

    a = np.asarray(crank_angle)
    da = np.diff(a)
    wraps = np.where(da < -np.pi)[0] + 1

    valid = [wraps[0]]
    for s in wraps[1:]:
        if s - valid[-1] >= min_cycle_frames:
            valid.append(s)

    return np.array(valid)


def normalize_cycles_by_crank(signals, crank_angle, starts, n_points=360):

    signals = np.asarray(signals)
    a = np.unwrap(crank_angle)

    angle_grid = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    cycles = []

    for i in range(len(starts)-1):

        i0, i1 = starts[i], starts[i+1]

        seg_a = a[i0:i1] - a[i0]
        seg_s = signals[:, i0:i1]

        phi = np.mod(seg_a, 2*np.pi)
        order = np.argsort(phi)

        phi = phi[order]
        seg_s = seg_s[:, order]

        interp = np.zeros((signals.shape[0], n_points))

        for m in range(signals.shape[0]):
            interp[m] = np.interp(angle_grid, phi, seg_s[m])

        cycles.append(interp)

    cycles = np.stack(cycles, axis=1)
    return cycles, angle_grid



def normalize_1d_cycles_by_crank(sig_1d, crank_angle, cycle_starts, n_points=360):
    sig_1d = np.asarray(sig_1d, float).reshape(1, -1)
    cyc, angle_grid = normalize_cycles_by_crank(sig_1d, crank_angle, cycle_starts, n_points=n_points)
    return cyc[0], angle_grid  # (n_cycles, n_points), (n_points,)


# ============================================================
# Stats par mode (activations + forces) avec mêmes cycles (crank)
# ============================================================
def compute_mode_cycle_stats_crank(
    q, activations, forces, tau, v_musc, crank_angle,
    min_cycle_frames=30,
    n_points=360,
):
    """
    q           : (nbQ, n_frames)   (optionnel pour plots/diagnostic)
    activations : (nbMuscles, n_frames)
    forces      : (nbMuscles, n_frames)
    crank_angle : (n_frames,)
    """
    q = np.asarray(q, float)
    activations = np.asarray(activations, float)
    forces = np.asarray(forces, float)
    tau = np.asarray(tau, float)
    v_musc = np.asarray(v_musc, float)
    crank_angle = np.asarray(crank_angle, float)

    if q.shape[1] != activations.shape[1] or q.shape[1] != forces.shape[1] or q.shape[1] != crank_angle.shape[0]:
        raise ValueError("q, activations, forces, crank_angle doivent avoir le même n_frames.")
    if activations.shape != forces.shape:
        raise ValueError("activations et forces doivent avoir la même shape.")

    starts = detect_cycles_from_crank(crank_angle, min_cycle_frames=min_cycle_frames)

    act_cycles, angle_grid = normalize_cycles_by_crank(activations, crank_angle, starts, n_points=n_points)
    frc_cycles, _          = normalize_cycles_by_crank(forces,      crank_angle, starts, n_points=n_points)
    tau_cycles, _          = normalize_cycles_by_crank(tau, crank_angle, starts, n_points=n_points)
    v_musc_cycles, _ = normalize_cycles_by_crank(v_musc, crank_angle, starts, n_points=n_points)
    q_cycles2, _ = normalize_cycles_by_crank(q, crank_angle, starts, n_points=n_points)
    q_cycles2 = np.rad2deg(q_cycles2)
    # (optionnel) normaliser aussi un q de réf pour vérifier l'alignement
    # ici on prend q[14] par défaut si dispo
    q_ref = q[14, :] if q.shape[0] > 14 else q[0, :]
    q_cycles, _ = normalize_1d_cycles_by_crank(q_ref, crank_angle, starts, n_points=n_points)


    stats = {
        "starts": starts,
        "angle_grid": angle_grid,
        "q_cycles": q_cycles,
        "q_mean": q_cycles2.mean(axis=1),
        "q_std": q_cycles2.std(axis=1),
        "act_cycles": act_cycles,
        "frc_cycles": frc_cycles,
        "act_mean": act_cycles.mean(axis=1),
        "act_std":  act_cycles.std(axis=1),
        "frc_mean": frc_cycles.mean(axis=1),
        "frc_std":  frc_cycles.std(axis=1),
        "tau_cycles": tau_cycles,
        "tau_mean": tau_cycles.mean(axis=1),
        "tau_std": tau_cycles.std(axis=1),
        "v_musc_mean": v_musc_cycles.mean(axis=1),
        "v_musc_std": v_musc_cycles.std(axis=1)
    }
    return stats


# ============================================================
# Plot helpers
# ============================================================
def plot_q_alignment_angle(q_cycles_con, q_cycles_ecc, angle_grid, title):
    x_deg = (np.rad2deg(angle_grid) % 360)

    plt.figure(figsize=(12, 4))
    for c in q_cycles_con:
        plt.plot(x_deg, c, alpha=0.15)
    for c in q_cycles_ecc:
        plt.plot(x_deg, c, alpha=0.15)

    plt.plot(x_deg, q_cycles_con.mean(axis=0), linewidth=2.5, label=f"con (moy, N={q_cycles_con.shape[0]})")
    plt.plot(x_deg, q_cycles_ecc.mean(axis=0), linewidth=2.5, label=f"ecc (moy, N={q_cycles_ecc.shape[0]})")

    plt.title(title)
    plt.xlabel("Angle pédalier (deg)")
    plt.ylabel("q_ref (a.u.)")
    plt.xlim(0, 360)
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()


def plot_crank_with_starts(crank_angle, starts, title):
    a = np.asarray(crank_angle, float)
    plt.figure(figsize=(12, 3))
    plt.plot(a, label="crank_angle (rad)")
    plt.plot(starts, a[starts], "ro", label="cycle starts")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_with_mode(ax, x, y, v, color, label=None):
    """
    Trace une courbe avec style dépendant de v (point par point)
    color = fixe (CON vs ECC)
    """

    def get_style(val):
        if val < -0.01:
            return "--"   # concentrique
        elif val > 0.01:
            return ":"   # excentrique
        else:
            return "-"    # isométrique

    current_style = get_style(v[0])
    start_idx = 0

    for i in range(1, len(x)):
        new_style = get_style(v[i])

        if new_style != current_style:
            ax.plot(
                x[start_idx:i],
                y[start_idx:i],
                linestyle=current_style,
                color=color,
                label=label if start_idx == 0 else None
            )
            start_idx = i
            current_style = new_style

    # dernier segment
    ax.plot(
        x[start_idx:],
        y[start_idx:],
        linestyle=current_style,
        color=color,
        label=label if start_idx == 0 else None
    )

def plot_grid_mean_std_angle_multi(
    means_list, stds_list,
    labels, colors,
    muscle_names, angle_grid,
    y_label, suptitle,
    show_mode=False,
    v_musc_list=None,
):
    n_muscles, n_points = means_list[0].shape
    x_deg = (np.rad2deg(angle_grid) % 360)

    ncols = 5
    nrows = int(np.ceil(n_muscles / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows), sharex=True)
    axes = np.atleast_1d(axes).flatten()

    for m in range(n_muscles):
        ax = axes[m]

        for i, (mean, std, label, color) in enumerate(zip(means_list, stds_list, labels, colors)):

            if show_mode and v_musc_list is not None:
                plot_with_mode(
                    ax,
                    x_deg,
                    mean[m],
                    v_musc_list[i][m],
                    color=color,
                    label=label if m == 0 else None
                )
            else:
                ax.plot(
                    x_deg,
                    mean[m],
                    color=color,
                    label=label if m == 0 else None
                )

            ax.fill_between(
                x_deg,
                mean[m] - std[m],
                mean[m] + std[m],
                alpha=0.2,
                color=color
            )

        title = muscle_names[m] if muscle_names is not None else f"muscle_{m}"
        ax.set_title(title)

        if m // ncols == nrows - 1:
            ax.set_xlabel("Angle pédalier (deg)")
        else:
            ax.set_xlabel("")

        ax.set_ylabel(y_label)
        ax.set_xlim(0, 360)
        ax.grid(True, alpha=0.3)

    # remove empty axes
    for k in range(n_muscles, len(axes)):
        fig.delaxes(axes[k])

    # légende
    from matplotlib.lines import Line2D

    cond_legend = [
        Line2D([0], [0], color=c, lw=2, label=l)
        for c, l in zip(colors, labels)
    ]

    style_legend = [
        Line2D([0], [0], color="black", linestyle="--", lw=2, label="Mode concentrique"),
        Line2D([0], [0], color="black", linestyle="-", lw=2, label="Mode isométrique"),
        Line2D([0], [0], color="black", linestyle=":", lw=2, label="Mode excentrique")
    ]

    handles = cond_legend + (style_legend if show_mode else [])

    fig.legend(
        handles=handles,
        loc="lower right",
        bbox_to_anchor=(0.98, 0.02),
        frameon=True,
        fontsize=11,
    )

    fig.subplots_adjust(top=0.92, bottom=0.08, hspace=0.35, wspace=0.25)
    fig.suptitle(suptitle, fontsize=14)

    plt.show()




def plot_tau_grid_layout(
    dof_name,               # liste de tous les dofs/muscles disponibles            # indices de cycles pour extraction
    tau_mean_list,          # liste [tau_mean_con, tau_mean_ecc], shape: n_dof x n_points
    tau_std_list,           # liste [tau_std_con, tau_std_ecc], shape: n_dof x n_points
    layout,                 # dict segment -> list of (dof, titre)
    labels=None,            # ex: ["Concentrique", "Excentrique"]
    colors=None,            # ex: ["royalblue", "tomato"]
    n_points=360,
    ylabel="Torque (N·m)"
):
    if labels is None:
        labels = [f"Method {i+1}" for i in range(len(tau_mean_list))]
    if colors is None:
        colors = [None] * len(tau_mean_list)

    segments = list(layout.keys())
    n_rows = len(segments)
    n_cols = max(len(layout[s]) for s in segments)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11.69, 8.27), sharex=True)
    plt.subplots_adjust(left=0.07, right=0.98, top=0.93, bottom=0.08, wspace=0.25, hspace=0.35)

    if n_rows == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes[:, None]

    x = np.linspace(0, 360, n_points)

    for r, seg in enumerate(segments):
        dofs = layout[seg]
        for c in range(n_cols):
            ax = axes[r, c]

            # case vide si moins de colonnes
            if c >= len(dofs):
                ax.axis("off")
                continue

            dof, title = dofs[c]
            if dof not in dof_name:
                ax.set_title(f"{title}\n(MISSING: {dof})", fontsize=12)
                ax.axis("off")
                continue

            idx = dof_name.index(dof)

            for tau_mean, tau_std, lab, col in zip(tau_mean_list, tau_std_list, labels, colors):
                cyc_mean = tau_mean[idx]  # déjà moyenné sur cycles
                cyc_std  = tau_std[idx]
                ax.plot(x, cyc_mean, lw=2, label=lab, color=col)
                ax.fill_between(x, cyc_mean - cyc_std, cyc_mean + cyc_std, alpha=0.15, color=col)

            ax.set_title(title, fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.tick_params(labelsize=10)

            if c == 0:
                ax.set_ylabel(f"{seg}\n{ylabel}")
            else:
                ax.set_ylabel("")

            if r == n_rows - 1:
                ax.set_xlabel("Crank angle (°)")

    # Une seule légende globale
    handles, leg_labels = axes[0, 0].get_legend_handles_labels()
    if len(handles) > 0:
        fig.legend(handles, leg_labels, loc="lower right", frameon=False)

    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.show()

# ============================================================
# MAIN (à brancher sur tes arrays)
# ============================================================
if __name__ == "__main__":
    MIN_CYCLE_FRAMES = 30
    N_POINTS = 360
    stats_all = {}

    for PUISSANCE in PUISSANCES:

        if ESSAI == "Collecte_18_03":
            if PUISSANCE == "40":
                START = 2000
                END = 5000
            elif PUISSANCE == "60":
                START = 2000
                END = 5000
            elif PUISSANCE == "80":
                START = 1500
                END = 4000

        BASE = f"/Users/leo/Desktop/Projet/{ESSAI}/{MODE}_{PUISSANCE}W"

        q = np.load(f"{BASE}/q_inverse_kinematic.npy")[:, START:END]
        act = np.load(f"{BASE}/muscle_activations_nonlinear.npy")[:, :END - START]
        frc = np.load(f"{BASE}/muscles_forces.npy")[:, :END - START]
        tau = np.load(f"{BASE}/tau_inverse_dynamic.npy")[:, START:END]
        crank = np.load(f"{BASE}/crank_angle.npy")[START:END]
        v_musc = np.load(f"{BASE}/vitesse_musculaire.npy")[:, :END - START]

        crank, q, act, frc, tau, v_musc = ensure_forward_rotation(
            crank, q, act, frc, tau, v_musc
        )

        stats = compute_mode_cycle_stats_crank(
            q, act, frc, tau, v_musc, crank,
            min_cycle_frames=MIN_CYCLE_FRAMES,
            n_points=N_POINTS,
        )

        stats_all[PUISSANCE] = stats

    model = biorbd.Model(f"/Users/leo/Desktop/Projet/{ESSAI}/model_{ESSAI}.bioMod")
    muscle_names = [model.muscleNames()[i].to_string() for i in range(int(model.nbMuscles()))]
    dof_name = [model.nameDof()[i].to_string() for i in range(int(model.nbDof()))]
    #
    LAYOUT = {
            "Clavicle": [
                ("thorax_offset_sternoclavicular_left_r1_RotX", "Pro/retraction"),
                ("thorax_offset_sternoclavicular_left_r2_RotY", "Depression/Elevation"),
            ],
            "Scapula": [
                ("scapula_left_rotation_transform_RotX", "Pro/retraction"),
                ("scapula_left_rotation_transform_RotY", "Lat/med rotation"),
                ("scapula_left_rotation_transform_RotZ", "Tilt"),
            ],
            "Humerus": [
                ("scapula_left_offset_shoulder_left_plane_RotX", "Plane of elevation"),
                ("scapula_left_offset_shoulder_left_ele_RotY", "Elevation"),
                ("scapula_left_offset_shoulder_left_rotation_RotZ", "Axial rotation"),
            ],
            "Forearm": [
                ("humerus_left_offset_elbow_left_flexion_RotZ", "Flexion/extension"),
                ("ulna_left_offset_pro_sup_left_RotY", "Pronation/supination"),
            ],
        }

    means = [stats_all[p]["act_mean"] for p in PUISSANCES]
    stds = [stats_all[p]["act_std"] for p in PUISSANCES]
    vlist = [stats_all[p]["v_musc_mean"] for p in PUISSANCES]

    plot_grid_mean_std_angle_multi(
        means, stds,
        labels=[f"{p}W" for p in PUISSANCES],
        colors=["royalblue", "orange", "green"],
        muscle_names=muscle_names,
        angle_grid=stats_all[PUISSANCES[0]]["angle_grid"],
        y_label="Activation",
        suptitle=f"Activations — {MODE} — comparaison puissances",
        show_mode=True,
        v_musc_list=vlist,
    )

    means = [stats_all[p]["frc_mean"] for p in PUISSANCES]
    stds = [stats_all[p]["frc_std"] for p in PUISSANCES]

    plot_grid_mean_std_angle_multi(
        means, stds,
        labels=[f"{p}W" for p in PUISSANCES],
        colors=["royalblue", "orange", "green"],
        muscle_names=muscle_names,
        angle_grid=stats_all[PUISSANCES[0]]["angle_grid"],
        y_label="Force musculaire (N)",
        suptitle=f"Forces — {MODE} — comparaison puissances",
    )

    means = [stats_all[p]["v_musc_mean"] for p in PUISSANCES]
    stds = [stats_all[p]["v_musc_std"] for p in PUISSANCES]

    plot_grid_mean_std_angle_multi(
        means, stds,
        labels=[f"{p}W" for p in PUISSANCES],
        colors=["royalblue", "orange", "green"],
        muscle_names=muscle_names,
        angle_grid=stats_all[PUISSANCES[0]]["angle_grid"],
        y_label="Vitesse musculaire (m/s)",
        suptitle=f"Vitesses — {MODE} — comparaison puissances",
    )

    plot_tau_grid_layout(
        dof_name=dof_name,
        tau_mean_list=[stats_all[p]["q_mean"] for p in PUISSANCES],
        tau_std_list=[stats_all[p]["q_std"] for p in PUISSANCES],
        layout=LAYOUT,
        labels=[f"{p}W" for p in PUISSANCES],
        colors=["royalblue", "orange", "green"],
        n_points=N_POINTS,
        ylabel="Joint angle (°)"
    )

    plot_tau_grid_layout(
        dof_name=dof_name,
        tau_mean_list=[stats_all[p]["tau_mean"] for p in PUISSANCES],
        tau_std_list=[stats_all[p]["tau_std"] for p in PUISSANCES],
        layout=LAYOUT,
        labels=[f"{p}W" for p in PUISSANCES],
        colors=["royalblue", "orange", "green"],
        n_points=N_POINTS,
        ylabel="Torque (N·m)"
    )