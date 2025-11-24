import os.path
from biosiglive import load, save, OfflineProcessing
import numpy as np
import matplotlib.pyplot as plt
from pyomeca import Analogs, Markers
import glob
from pathlib import Path

#prefix = "/mnt/shared/" if os.name == "posix" else "Q:\\"
#pref_bis = "/media/mickael/566A33906A336BBD/" if os.name == "posix" else "C:\\"
#sep = os.sep

#emg_path_signal = "/home/mickael/Documents/Leo/EMG/emg_signal/muscle_activation.c3d"
#emg_path_mcv = "/home/mickael/Documents/Leo/EMG/emg_signal/mcv.c3d"



if __name__ == '__main__':
    participants = ["P0"]
    processed_data = "/processed_data"
    data_files = "/Users/leo/Desktop/Projet/Données/Test_1.c3d"
    emg_names = ['Delt_ant.IM EMG1', 'Delt_med.IM EMG2',
      'Delt_post.IM EMG3',
       'Trap_med.IM EMG4', 'Biceps.IM EMG5', 'Triceps.IM EMG6',
       'Pec.IM EMG7', 'Brachio.IM EMG8', 'Trigger.1']
    markers_names = ['Ster', 'Xiph', 'C7', 'T10', 'Clav_SC', 'Clav_Mid', 'Clav_AC', 'Scap_AA', 'Scap_TS',
       'Scap_IA', 'Delt', 'ArmI', 'EpicI', 'EpicM', 'Elbow', 'LArmI',
       'StylR', 'StylU', 'Hand_Top', 'Little_Base', 'Index_Base']
    #markers_cluster = ['M1', 'M2', 'M3', 'scapaa', 'scapts', 'scapia', 'slaa', 'slts', 'slai']

    for p, part in enumerate(participants):
        try:
            files = glob.glob(data_files)
        except:
            files = glob.glob(data_files)
        mvc_trials = [file for file in files if "sprint" in file]
        mvc_data = [Analogs.from_c3d(filename=file, usecols=emg_names).values for file in mvc_trials]
        mvc_mat = np.append(mvc_data[0], mvc_data[1], axis=1)
        mvc = OfflineProcessing.compute_mvc(mvc_data[0].shape[0], mvc_trials=mvc_mat, window_size=2160).tolist()
        for f, file in enumerate(files):
            if os.path.isfile(f"{processed_data}/{part}/{Path(file).stem}_processed.bio"):
                continue
            if "sprint" in file:
                continue
            markers_data = Markers.from_c3d(filename=file)
            markers_names_tmp = markers_data.channel.values
            if "ster" in markers_names_tmp:
                is_emg = True
                final_markers_names = markers_names

            else:
                print(f"error while loading markers on file {file}.")
                continue
            analog_data = Analogs.from_c3d(filename=file, usecols=emg_names).values
            trigger = analog_data[-1, :]
            trigger_values = np.argwhere(trigger[::18] > 1.5)
            if len(trigger_values) == 0:
                start_idx = 0
                end_idx = trigger[::18].shape[0]
            else:
                start_idx = int(trigger_values[0][0])
                try:
                    end_idx = int(trigger_values[int(np.argwhere(trigger_values > start_idx + 200)[0][0])])
                except:
                    end_idx = trigger[::18].shape[0]
            trigger_idx = [start_idx, end_idx]
            markers_vicon = markers_data[:, :, trigger_idx[0]:trigger_idx[1]]
            if is_emg:
                emg = analog_data[:-1, :]
                emg_proc = OfflineProcessing(data_rate=2160).process_emg(emg,
                                                                             moving_average=False,
                                                                             low_pass_filter=True,
                                                                             normalization=True,
                                                                              mvc_list=mvc)
                emg_proc = emg_proc[:, trigger_idx[0]*18:trigger_idx[1]*18]
                emg = emg[:, trigger_idx[0]*18:trigger_idx[1]*18]
            else:
                emg_proc, emg = [], []

            dic_to_save = {"file_name": file,
                           "markers": markers_vicon,
                           "raw_emg": emg,
                            "emg_proc": emg_proc,
                           "markers_names": final_markers_names,
                           "emg_names": emg_names,
                           "vicon_rate": 120, "emg_rate": 2160,
                           "mvc": mvc}
            save(dic_to_save, f"{processed_data}/{part}/{Path(file).stem}_processed.bio", add_data=True)
            print(f"file {part}/{Path(file).stem}_processed.bio saved")