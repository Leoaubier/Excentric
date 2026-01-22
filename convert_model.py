# conda env musculo

from biobuddy import BiomechanicalModelReal

scaled_osim_path = "/Users/leo/Desktop/Projet/modele_opensim/wu_bras_gauche_seth_left_Sidonie_V3.osim"
scaled_biomod_path = ("/Users/leo/Desktop/Projet/modele_opensim/wu_bras_gauche_seth_left_Sidonie_V3.bioMod")

model = BiomechanicalModelReal().from_osim(scaled_osim_path)
model.to_biomod(scaled_biomod_path)

print("Modèle BioMod scalé écrit dans :", scaled_biomod_path)