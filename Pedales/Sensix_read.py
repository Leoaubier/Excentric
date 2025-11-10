#from biosiglive import save, load, OfflineProcessing
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import csv
import biorbd
from pathlib import Path
#import bioviz
from scipy.interpolate import interp1d

prefix = "/mnt/shared/" if os.name == "posix" else "Q:\\"
pref_bis = "/media/mickael/566A33906A336BBD/" if os.name == "posix" else "C:\\"
sep = os.sep

sensix_path = '/home/mickael/Documents/Leo/Sensix/sensix_try_2_001.lvm'


def to_dic(all_data_int):
    dic_data = {"time": all_data_int[0, :],
                "LFX": all_data_int[1, :],
                "LFY": all_data_int[2, :],
                "LFZ": all_data_int[3, :],
                "LMX": all_data_int[4, :],
                "LMY": all_data_int[5, :],
                "LMZ": all_data_int[6, :],
                "RFX": all_data_int[7, :],
                "RFY": all_data_int[8, :],
                "RFZ": all_data_int[9, :],
                "RMX": all_data_int[10, :],
                "RMY": all_data_int[11, :],
                "RMZ": all_data_int[12, :],
                "crank_angle": all_data_int[15, :],
                "right_pedal_angle": all_data_int[14, :],
                "left_pedal_angle": all_data_int[13, :],
                }
    return dic_data


def read_sensix_files():
    pass

def express_forces_in_global(crank_angle, f_ext): #se mettre dans le referentiel du pedalier
    crank_angle = crank_angle
    Roty = np.array([[np.cos(crank_angle), 0, np.sin(crank_angle)],
                     [0, 1, 0],
                     [-np.sin(crank_angle), 0, np.cos(crank_angle)]])
    return Roty @ f_ext


def smooth_angle(key, dic_data):
    start_cycle = False
    for i in range(dic_data[key].shape[0]):
        if dic_data[key][i] < 0.1:
            start_cycle = True
        elif dic_data[key][i] > 6:
            start_cycle = False
        if start_cycle and dic_data[key][i] > 0.1 and dic_data[key][i] < 6:
            dic_data[key][i] = 0
    return dic_data[key]


if __name__ == '__main__':
    all_data = []
    with open(sensix_path, 'r') as f:
        csvreader = csv.reader(f, delimiter='\n')
        for row in csvreader:
            all_data.append(np.array(row[0].split("\t")))
    all_data = np.array(all_data, dtype=float).T

    dic_data = to_dic(all_data)
    for key in ["left_pedal_angle", "right_pedal_angle", "crank_angle"]:
        dic_data[key] = smooth_angle(key, dic_data)

    for i in range(all_data.shape[1]):
        crank_angle = dic_data["crank_angle"][i]
        left_angle = dic_data["left_pedal_angle"][i]
        right_angle = dic_data["right_pedal_angle"][i]
        force_vector_l = [dic_data["LFX"][i], dic_data["LFY"][i], dic_data["LFZ"][i]]
        force_vector_r = [dic_data["RFX"][i], dic_data["RFY"][i], dic_data["RFZ"][i]]
        force_vector_l = express_forces_in_global(-left_angle, force_vector_l)
        force_vector_r = express_forces_in_global(-right_angle, force_vector_r)
        force_vector_l = express_forces_in_global(crank_angle, force_vector_l)
        force_vector_r = express_forces_in_global(crank_angle, force_vector_r)
        dic_data["LFX"][i] = force_vector_l[0]
        dic_data["LFY"][i] = force_vector_l[1]
        dic_data["LFZ"][i] = force_vector_l[2]
        dic_data["RFX"][i] = force_vector_r[0]
        dic_data["RFY"][i] = force_vector_r[1]
        dic_data["RFZ"][i] = force_vector_r[2]
    #save(dic_data, f"/home/mickael/Documents/Leo/Sensix/sensix_try_2_001.bio")
    print("file :.bio saved")
    print(dic_data)
    fig, ax = plt.subplots(figsize=(10, 4), dpi=120)

    ax.plot(dic_data["time"], dic_data["LFX"], label="Fx gauche – pédalier", lw=1.8)
    #ax.plot(dic_data_pedale["time"], dic_data_pedale["LFX"], label="Fx gauche – pédale", lw=0.1)

    ax.set_xlabel("Temps (s)")
    ax.set_ylabel("Force X (N)")
    ax.set_title("Comparaison Fx gauche (pédalier vs pédale)")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(frameon=False, ncol=2)


    fig.tight_layout()
    plt.show()

