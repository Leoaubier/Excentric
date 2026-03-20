# conda env musculo

from biobuddy import BiomechanicalModelReal

scaled_osim_path = "/Users/leo/Desktop/Projet/Collecte_18_03/model_Collecte_18_03_V2.osim"
scaled_biomod_path = ("/Users/leo/Desktop/Projet/Collecte_18_03/model_Collecte_18_03.bioMod")

model = BiomechanicalModelReal().from_osim(scaled_osim_path)
model.to_biomod(scaled_biomod_path)

print("Modèle BioMod scalé écrit dans :", scaled_biomod_path)