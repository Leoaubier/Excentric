import pandas as pd

lvm_file = "/Users/leo/Desktop/Projet/Données/donnes_pedales_test_1.lvm"
mot_file = "/Users/leo/Desktop/Projet/Données/donnes_pedales_test_1.mot"

# Charge fichier LVM en sautant lignes d'entête
df = pd.read_csv(lvm_file, sep="\t", comment="#", header=None)

# Colonnes selon PDF (pages 27–28)
df.columns = [
    "time",
    "L_Fx", "L_Fy", "L_Fz",
    "L_Mx", "L_My", "L_Mz",
    "R_Fx", "R_Fy", "R_Fz",
    "R_Mx", "R_My", "R_Mz",
    "encoder_l", "encoder_r",
    "crank", "meta_donnees"
]

# Construction du fichier MOT
mot = pd.DataFrame({
    "time": df["time"],

    # Left pedal
    "L_ground_force_vx": df["L_Fx"],
    "L_ground_force_vy": df["L_Fy"],
    "L_ground_force_vz": df["L_Fz"],

    "L_ground_force_px": 0,
    "L_ground_force_py": 0,
    "L_ground_force_pz": 0,  # supposé plan

    "L_ground_torque_x": df["L_Mx"],
    "L_ground_torque_y": df["L_My"],
    "L_ground_torque_z": df["L_Mz"],

    # Right pedal
    "R_ground_force_vx": df["R_Fx"],
    "R_ground_force_vy": df["R_Fy"],
    "R_ground_force_vz": df["R_Fz"],

    "R_ground_force_px": 0,
    "R_ground_force_py": 0,
    "R_ground_force_pz": 0,

    "R_ground_torque_x": df["R_Mx"],
    "R_ground_torque_y": df["R_My"],
    "R_ground_torque_z": df["R_Mz"],

    "L_encoder": df["encoder_l"],
    "R_encoder": df["encoder_r"],
    "Crank": df["crank"]
})

# Écriture du fichier MOT
with open(mot_file, "w") as f:
    f.write("name forces\n")
    f.write("datacolumns {}\n".format(mot.shape[1]))
    f.write("datarows {}\n".format(mot.shape[0]))
    f.write("range {} {}\n".format(df.time.min(), df.time.max()))
    f.write("endheader\n")
mot.to_csv(mot_file, sep="\t", index=False, mode='a')

print("✔ MOT file generated:", mot_file)
