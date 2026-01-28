import sys
import time
import uuid
import multiprocessing as mp
from dataclasses import dataclass
from typing import List

import numpy as np
import biorbd


# ============================================================
# Data exchanged between UI -> viewer
# ============================================================
@dataclass
class Params:
    dof_idx: int                 # index in generalized coordinates/torques space
    activations: List[float]     # len = nbMuscles
    stop: bool = False


# ============================================================
# Helpers (robust conversions)
# ============================================================
def to_np(x):
    if hasattr(x, "to_array"):
        return np.array(x.to_array())
    return np.asarray(x)


def get_dof_names(model: biorbd.Model):
    """
    Retourne une liste de noms pour chaque q (nbQ).
    Compatible avec plusieurs versions biorbd.
    """
    nb_q = model.nbQ()
    names = []
    for i in range(nb_q):
        name = None

        # 1) nameDof(i)
        if hasattr(model, "nameDof"):
            try:
                v = model.nameDof(i)
                name = v.to_string() if hasattr(v, "to_string") else str(v)
            except Exception:
                pass

        # 2) dofNames() / nameQ(i) variantes
        if name is None:
            for attr in ("nameQ", "dofNames", "qNames", "dof_names"):
                if hasattr(model, attr):
                    try:
                        obj = getattr(model, attr)
                        if callable(obj):
                            v = obj(i)  # ex nameQ(i)
                            name = v.to_string() if hasattr(v, "to_string") else str(v)
                        else:
                            v = obj[i]
                            name = v.to_string() if hasattr(v, "to_string") else str(v)
                        break
                    except Exception:
                        pass

        if not name or name.strip() == "":
            name = f"q{i}"

        names.append(name)
    return names


# ============================================================
# Viewer process (pyorerun + rerun + compute)
# ============================================================
def viewer_process(model_path: str, q_params: mp.Queue):
    import threading
    import rerun as rr
    from pyorerun import LiveModelAnimation

    try:
        model = biorbd.Model(model_path)
        nb_q = model.nbQ()
        nb_mus = model.nbMuscles()
        qdot = np.zeros((nb_q,), dtype=float)

        rr.init("muscle_torque_explorer", spawn=True, recording_id=uuid.uuid4())

        # IMPORTANT: on laisse LiveModelAnimation gérer ses propres contrôles Q
        # (puisqu’on a retiré les sliders Q du Qt)
        animation = LiveModelAnimation(model_path, with_q_charts=False)

        muscles_obj = model.muscles()
        if isinstance(muscles_obj, tuple):
            muscles_obj = next((x for x in muscles_obj if hasattr(x, "muscularJointTorque")), None)

        # current params (no q here anymore)
        cur = Params(dof_idx=0, activations=[0.0] * nb_mus, stop=False)

        def build_states(acts: np.ndarray):
            return [biorbd.State(float(a), float(a)) for a in acts]

        def rr_set_frame(frame: int):
            if hasattr(rr, "set_time_sequence"):
                rr.set_time_sequence("frame", frame)

        def worker():
            nonlocal cur
            frame = 0
            while True:
                # Drain queue, keep latest
                try:
                    while True:
                        cur = q_params.get_nowait()
                except Exception:
                    pass

                if getattr(cur, "stop", False):
                    return

                # --- Read q directly from LiveModelAnimation ---
                # (this is what the built-in Q window modifies)
                q = None
                if hasattr(animation, "_q"):
                    q = np.asarray(animation._q, dtype=float).reshape((-1,))
                elif hasattr(animation, "q"):
                    q = np.asarray(animation.q, dtype=float).reshape((-1,))

                if q is None or q.shape[0] != nb_q:
                    q = np.zeros((nb_q,), dtype=float)

                dof = int(cur.dof_idx)
                dof = max(0, min(dof, nb_q - 1))

                acts = np.asarray(cur.activations, dtype=float).reshape((-1,))
                if acts.shape[0] != nb_mus:
                    acts = np.zeros((nb_mus,), dtype=float)

                # --- moment arms (all muscles for selected dof) ---
                J = to_np(model.musclesLengthJacobian(q))
                r_all = -np.asarray(J[:, dof], dtype=float).reshape((-1,))

                # --- total torque with all activations ---
                states_all = build_states(acts)
                if hasattr(model, "muscularJointTorque"):
                    tau_vec = model.muscularJointTorque(states_all, q, qdot)
                else:
                    tau_vec = muscles_obj.muscularJointTorque(states_all, q, qdot)

                tau_arr = to_np(tau_vec).reshape((-1,))
                tau_total = float(tau_arr[dof])

                rr_set_frame(frame)
                rr.log("analysis/tau_total", rr.Scalar(tau_total))
                rr.log("analysis/dof_idx", rr.Scalar(float(dof)))
                rr.log("analysis/q_selected", rr.Scalar(float(q[dof])))

                # --- per-muscle contribution ---
                eps = 1e-3
                active_ids = np.where(acts > eps)[0]

                for m in active_ids:
                    acts_one = np.zeros_like(acts)
                    acts_one[m] = acts[m]
                    states_one = build_states(acts_one)

                    if hasattr(model, "muscularJointTorque"):
                        tau_m_vec = model.muscularJointTorque(states_one, q, qdot)
                    else:
                        tau_m_vec = muscles_obj.muscularJointTorque(states_one, q, qdot)

                    tau_m = float(to_np(tau_m_vec).reshape((-1,))[dof])
                    r_m = float(r_all[m])

                    rr.log(f"analysis/muscles/{m:02d}/activation", rr.Scalar(float(acts[m])))
                    rr.log(f"analysis/muscles/{m:02d}/tau", rr.Scalar(tau_m))
                    rr.log(f"analysis/muscles/{m:02d}/moment_arm", rr.Scalar(r_m))

                frame += 1
                time.sleep(0.10)

        threading.Thread(target=worker, daemon=True).start()
        animation.rerun()

    except Exception:
        import traceback
        print("\n=== VIEWER_PROCESS CRASH ===")
        traceback.print_exc()
        print("=== END CRASH ===\n")
        time.sleep(2)
        raise


# ============================================================
# UI (PyQt5) - only DoF + muscles
# ============================================================
def pick_biomod_file() -> str | None:
    from PyQt5 import QtWidgets
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    path, _ = QtWidgets.QFileDialog.getOpenFileName(
        None,
        "Choisir un modèle .bioMod",
        "",
        "Biorbd models (*.bioMod);;All files (*)",
    )
    return path if path else None


def ui_main(model_path: str):
    from PyQt5 import QtWidgets, QtCore

    model = biorbd.Model(model_path)
    nb_q = model.nbQ()
    nb_mus = model.nbMuscles()

    # Muscle names
    muscle_names = []
    for i in range(nb_mus):
        try:
            muscle_names.append(model.muscle(i).name().to_string())
        except Exception:
            muscle_names.append(f"muscle_{i}")

    # DoF names (shown in dropdown)
    dof_names = get_dof_names(model)

    q_params = mp.Queue()

    # Start viewer in separate process (NOT daemon)
    p = mp.Process(target=viewer_process, args=(model_path, q_params))
    p.daemon = False
    p.start()

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    w = QtWidgets.QWidget()
    w.setWindowTitle("Muscles torque explorer (DoF + activations)")

    main_layout = QtWidgets.QVBoxLayout(w)

    # DoF choice
    main_layout.addWidget(QtWidgets.QLabel("DoF (tau) à analyser"))
    dof_cb = QtWidgets.QComboBox()
    dof_cb.addItems([f"{i:02d} - {dof_names[i]}" for i in range(nb_q)])
    main_layout.addWidget(dof_cb)

    # Muscles list with checkbox + slider
    main_layout.addWidget(QtWidgets.QLabel("Muscles à activer (checkbox + slider)"))

    mus_scroll = QtWidgets.QScrollArea()
    mus_scroll.setWidgetResizable(True)
    mus_container = QtWidgets.QWidget()
    mus_layout = QtWidgets.QVBoxLayout(mus_container)

    mus_checks = []
    mus_sliders = []
    mus_labels = []

    def make_mus_row(i):
        box = QtWidgets.QCheckBox(f"{i:02d} - {muscle_names[i]}")
        slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(0)
        slider.setEnabled(False)

        lbl = QtWidgets.QLabel("0.00")

        def on_check(state):
            slider.setEnabled(state == QtCore.Qt.Checked)
            if state != QtCore.Qt.Checked:
                slider.setValue(0)

        def on_slide():
            a = slider.value() / 100.0
            lbl.setText(f"{a:.2f}")

        box.stateChanged.connect(on_check)
        slider.valueChanged.connect(on_slide)
        on_slide()

        row = QtWidgets.QHBoxLayout()
        row.addWidget(box, 2)
        row.addWidget(slider, 3)
        row.addWidget(lbl, 0)

        wrap = QtWidgets.QWidget()
        wrap.setLayout(row)
        return box, slider, lbl, wrap

    for i in range(nb_mus):
        box, slider, lbl, wrap = make_mus_row(i)
        mus_checks.append(box)
        mus_sliders.append(slider)
        mus_labels.append(lbl)
        mus_layout.addWidget(wrap)

    mus_layout.addStretch(1)
    mus_scroll.setWidget(mus_container)
    main_layout.addWidget(mus_scroll, 1)

    # Controls
    btn_all_off = QtWidgets.QPushButton("All OFF")
    btn_all_on = QtWidgets.QPushButton("All ON (0.30)")
    btn_send_once = QtWidgets.QPushButton("Send now")

    def all_off():
        for i in range(nb_mus):
            mus_checks[i].setChecked(False)
            mus_sliders[i].setValue(0)

    def all_on():
        for i in range(nb_mus):
            mus_checks[i].setChecked(True)
            mus_sliders[i].setValue(30)

    btn_all_off.clicked.connect(all_off)
    btn_all_on.clicked.connect(all_on)

    btn_row = QtWidgets.QHBoxLayout()
    btn_row.addWidget(btn_all_off)
    btn_row.addWidget(btn_all_on)
    btn_row.addWidget(btn_send_once)
    main_layout.addLayout(btn_row)

    # ---------------- Sending params ----------------
    def collect_params() -> Params:
        dof = dof_cb.currentIndex()
        acts = []
        for i in range(nb_mus):
            if mus_checks[i].isChecked():
                acts.append(mus_sliders[i].value() / 100.0)
            else:
                acts.append(0.0)
        return Params(dof_idx=dof, activations=acts, stop=False)

    def push():
        q_params.put(collect_params())

    btn_send_once.clicked.connect(push)

    # Auto-send at 10 Hz
    timer = QtCore.QTimer()
    timer.timeout.connect(push)
    timer.start(100)

    # Monitor viewer alive
    alive_timer = QtCore.QTimer()
    def check_viewer():
        if not p.is_alive():
            print("⚠️ Viewer process dead. exitcode =", p.exitcode)
            alive_timer.stop()
    alive_timer.timeout.connect(check_viewer)
    alive_timer.start(500)

    w.resize(800, 700)
    w.show()
    app.exec()

    # Clean shutdown
    try:
        q_params.put(Params(dof_idx=0, activations=[0.0]*nb_mus, stop=True))
    except Exception:
        pass
    p.join(timeout=2)
    if p.is_alive():
        p.terminate()
        p.join(timeout=1)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)  # macOS

    model_path = sys.argv[1] if len(sys.argv) >= 2 else None
    if not model_path:
        model_path = pick_biomod_file()
        if not model_path:
            print("Aucun fichier sélectionné. Fin.")
            raise SystemExit(0)

    ui_main(model_path)
