import pandas as pd
from rdkit import Chem
from create_fingerprints import create_ecfp4_fingerprint, create_maccs_fingerprint

# Create dataframe from csv
df = pd.read_csv("./datasets/sider.csv")

# Extract SMILES column
df_molecules = pd.DataFrame(df["smiles"])

# Converting to molecules
df_molecules["mols"] = df_molecules["smiles"].apply(Chem.MolFromSmiles)
# df_molecules.to_csv("./datasets/df_molecules")

# Morgan Fingerprint (ECFP4)
ecfp_df = create_ecfp4_fingerprint(df_molecules, False)

# MACCS keys
maccs_df = create_maccs_fingerprint(df_molecules, False)

print(df_molecules)
print(ecfp_df)
print(maccs_df)
