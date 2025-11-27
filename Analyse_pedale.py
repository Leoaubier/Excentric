from pyomeca import Analogs
import pandas as pd
import matplotlib.pyplot as plt

mot_file = '/Users/leo/Desktop/Projet/Données/donnes_pedales_test_1.mot'

def read_mot_file(filepath):
    """Reads a .mot file and returns a pandas DataFrame."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Find the line where data starts (after 'endheader')
    data_start_index = -1
    for i, line in enumerate(lines):
        if 'endheader' in line.lower():
            data_start_index = i + 1
            break

    if data_start_index == -1:
        raise ValueError("Could not find 'endheader' in the .mot file.")

    # Read the data into a pandas DataFrame, assuming space-separated values
    # The first line after 'endheader' usually contains column headers
    df = pd.read_csv(filepath, sep='\t', skiprows=data_start_index, header=0)
    return df

# Example usage:
mot_data = read_mot_file(mot_file)
print(mot_data.head())


# Assuming 'mot_data' is your DataFrame from the previous step
# Example: Plotting 'time' vs 'some_variable'
plt.figure(figsize=(10, 6))
plt.plot(mot_data['time'], mot_data['L_ground_force_vx'], label='L_ground_force_vx')
plt.xlabel('Time (s)')
plt.ylabel('L_ground_force_vx (N)')
plt.title('L_ground_force_vx over Time')
plt.legend()
plt.grid(True)
#plt.show()

# You can plot multiple variables on the same graph:
plt.plot(mot_data['time'], mot_data['R_ground_force_vx'], label='L_ground_force_vx')
plt.legend()

fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
ax.plot(mot_data['Crank'], mot_data['L_ground_force_vx'], color='blue', linewidth=2)
ax.plot(mot_data['Crank'], mot_data['R_ground_force_vx'], color='orange', linewidth=2)
#ax.plot(mot_data['Crank'], mot_data['L_ground_force_vz'], color='red', linewidth=2)

plt.show()
