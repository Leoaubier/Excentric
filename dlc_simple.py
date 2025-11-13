import os
import cv2
import numpy as np
import deeplabcut
import pandas as pd

# === PARAMÈTRES À MODIFIER ===
# Chemin vers ton projet DeepLabCut (celui contenant config.yaml)
project_path = "/chemin/vers/ton/projet-DLC"

# Dossier contenant les images de profondeur 16 bits (.png)
depth_folder = "/chemin/vers/images_depth16/"

# Dossier temporaire pour stocker les images converties en 8 bits
converted_folder = os.path.join(depth_folder, "converted_8bit")
os.makedirs(converted_folder, exist_ok=True)

# Dossier de sortie pour les fichiers de résultats DLC
output_dir = os.path.join(project_path, "analyzed_depth")
os.makedirs(output_dir, exist_ok=True)

# === 1️⃣ Conversion des images 16 bits → 8 bits ===
print("🔄 Conversion des images 16 bits en 8 bits...")
for file in os.listdir(depth_folder):
    if file.lower().endswith(".png"):
        path_in = os.path.join(depth_folder, file)
        path_out = os.path.join(converted_folder, file)

        # Lecture en 16 bits
        depth_img = cv2.imread(path_in, cv2.IMREAD_UNCHANGED)

        if depth_img is None:
            print(f"⚠️ Erreur lecture {file}, ignorée.")
            continue

        # Normalisation sur 8 bits (0-255)
        normalized = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX)
        normalized_8bit = np.uint8(normalized)

        # Sauvegarde
        cv2.imwrite(path_out, normalized_8bit)

print("✅ Conversion terminée.")

# === 2️⃣ Application du modèle DeepLabCut ===
print("🧠 Application du modèle DeepLabCut...")
deeplabcut.analyze_images(
    project_path,
    [converted_folder],
    save_as_csv=True,
    destfolder=output_dir
)

# === 3️⃣ Extraction des positions de marqueurs ===
print("📊 Extraction des positions...")
csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
positions = []

for csv_file in csv_files:
    df = pd.read_csv(os.path.join(output_dir, csv_file), header=[1, 2])  # Multi-index (bodypart, coord)
    frame_data = df.iloc[0]
    marker_positions = {}

    for marker in df.columns.get_level_values(0).unique():
        x = frame_data[(marker, 'x')]
        y = frame_data[(marker, 'y')]
        marker_positions[marker] = {'x': float(x), 'y': float(y)}

    positions.append({
        'image': csv_file.replace('.csv', ''),
        'markers': marker_positions
    })

# === 4️⃣ Sauvegarde globale des positions ===
positions_df = pd.DataFrame([
    {'image': p['image'], **{f"{m}_{c}": v[c] for m, v in p['markers'].items() for c in ('x', 'y')}}
    for p in positions
])
positions_df.to_csv(os.path.join(output_dir, "positions_all_markers.csv"), index=False)

print("✅ Analyse terminée ! Résultats enregistrés dans :")
print(os.path.join(output_dir, "positions_all_markers.csv"))
