import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors, AllChem
from aux_functions import getMorganWithTry, tonumpyarraytolist

# pd.set_option('display.max_columns', 2)


# Create dataframe from csv
df = pd.read_csv("./datasets/sider.csv")

# Extract SMILES column
df_molecules = pd.DataFrame(df["smiles"])

# Converting to molecules
df_molecules["mols"] = df_molecules["smiles"].apply(Chem.MolFromSmiles)

# Morgan Fingerprint (ECFP4)
df_molecules["ECFP4"] = df_molecules["mols"].apply(getMorganWithTry).apply(tonumpyarraytolist)
print(df_molecules)
# print(type(df_molecules.loc[0, "ECFP4"]))

# New DF with one column for each ECFP bit
ecfp_df = df_molecules['ECFP4'].apply(pd.Series)
ecfp_df = ecfp_df.rename(columns=lambda x: 'ECFP4_' + str(x + 1))

# Write to csvs
df_molecules.to_csv("./datasets/df_molecules")
ecfp_df.to_csv("./datasets/ecfp4.csv")

print(ecfp_df)
