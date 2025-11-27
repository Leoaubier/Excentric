from biobuddy import BiomechanicalModelReal

scaled_osim_path = "/Users/leo/Desktop/Projet/modele_opensim/wu_bras_gauche_seth_left_Sidonie.osim"
scaled_biomod_path = "/Users/leo/Desktop/Projet/modele_opensim/wu_bras_gauche_seth_left_Sidonie.BioMod"

model = BiomechanicalModelReal().from_osim(scaled_osim_path)
model.to_biomod(scaled_biomod_path)

print("Modèle BioMod scalé écrit dans :", scaled_biomod_path)