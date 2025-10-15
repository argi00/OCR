import os
import random
from trdg.generators import GeneratorFromStrings

# === Étape 1 : Génération du texte synthétique ===
noms = ["Jean Dupont", "Marie Curie", "Ali Traoré", "Fatou Diop", "Luc Martin", "Aïcha Diallo"]
dates = ["14/10/2025", "03/09/2024", "25/12/2023", "11/01/2025", "09/06/2022"]
hemo = [str(round(random.uniform(11.0, 16.0), 1)) + " g/dL" for _ in range(20)]
gly = [str(round(random.uniform(3.5, 6.0), 1)) + " mmol/L" for _ in range(20)]
creat = [str(random.randint(60, 120)) + " µmol/L" for _ in range(20)]
chol = [str(round(random.uniform(1.5, 2.5), 1)) + " g/L" for _ in range(20)]

def generate_line():
    """Construit une ligne de texte simulant un rapport de laboratoire."""
    return f"Nom : {random.choice(noms)} | Date : {random.choice(dates)} | " \
           f"Hémoglobine : {random.choice(hemo)} | Glycémie à jeun : {random.choice(gly)} | " \
           f"Créatinine : {random.choice(creat)} | Cholestérol total : {random.choice(chol)}"

# Génère 7000 lignes de texte
all_texts = [generate_line() for _ in range(7000)]
print(f"Nombre total de lignes à générer : {len(all_texts)}")

# === Étape 2 : Génération des images synthétiques ===
output_dir = "./data/synthetic/images"
os.makedirs(output_dir, exist_ok=True)

# Utilisation d’un fond blanc (background_type=0) pour éviter l’erreur de chemin manquant
generator = GeneratorFromStrings(
    all_texts,
    count=len(all_texts),
    language="fr",
    blur=1,               # léger flou pour simuler un scan
    skewing_angle=3,      # texte légèrement incliné
    random_skew=True,
    background_type=0,    # fond blanc — pas besoin de fichiers externes
    size=64               # taille du texte
)

# === Étape 3 : Sauvegarde des images et labels ===
labels_path = "./data/synthetic/labels.txt"
with open(labels_path, "w", encoding="utf-8") as labels_file:
    for i, (img, lbl) in enumerate(generator):
        img_path = os.path.join(output_dir, f"synthetic_{i:05d}.jpg")
        img.save(img_path)
        labels_file.write(f"{img_path}\t{lbl}\n")

print("\n✅ Génération terminée avec succès !")
print(f"Images sauvegardées dans : {output_dir}")
print(f"Labels sauvegardés dans : {labels_path}")
