#musculo
from pathlib import Path
import vtk

GEOM_DIR = Path("/Users/leo/Desktop/Projet/modele_opensim/Geometry")

# Mets exactement les noms de fichiers utilisés (avec extension)
FILES = [
    "hat_spine.vtp",
    "thorax.vtp",
    "scapula_left.vtp",
    "clavicle_left.vtp",
    "humerus_left.vtp",
    "ulna_left.vtp",
    "radius_left.vtp",
]

OVERWRITE = False  # True = écrase les fichiers ; False = écrit *_triangulated.vtp

def triangulate_vtp(in_path: Path, out_path: Path):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(str(in_path))
    reader.Update()

    poly = reader.GetOutput()
    if poly is None:
        raise RuntimeError(f"Impossible de lire: {in_path}")

    # Triangulation
    tri = vtk.vtkTriangleFilter()
    tri.SetInputData(poly)
    tri.Update()

    # Nettoyage utile (évite certains artefacts)
    clean = vtk.vtkCleanPolyData()
    clean.SetInputData(tri.GetOutput())
    clean.Update()

    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(str(out_path))
    writer.SetInputData(clean.GetOutput())
    ok = writer.Write()
    if not ok:
        raise RuntimeError(f"Echec écriture: {out_path}")

def main():
    for f in FILES:
        in_path = GEOM_DIR / f
        if not in_path.exists():
            print(f"[SKIP] introuvable: {in_path}")
            continue

        if OVERWRITE:
            out_path = in_path
        else:
            out_path = in_path.with_name(in_path.stem + "_triangulated" + in_path.suffix)

        print(f"[OK] {in_path.name} -> {out_path.name}")
        triangulate_vtp(in_path, out_path)

    print("\nTerminé.")

if __name__ == "__main__":
    main()
