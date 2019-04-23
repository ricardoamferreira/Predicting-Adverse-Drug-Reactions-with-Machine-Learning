import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors, AllChem

def convert_numpy_to_list(np_array):
    return np_array.tolist()

def _getMorganWithTry(molecule):
    try:
        arr = np.zeros((1,))
        desc = rdMolDescriptors.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(str(molecule)), radius, 1024)
        DataStructs.ConvertToNumpyArray(desc, arr)
    except Exception as e:
        print(e)
        print('error ' + str(molecule))
        arr = np.nan
    return arr

# Create dataframe from csv
df = pd.read_csv("./datasets/sider.csv")

# Extract SMILES column
df_molecules = pd.DataFrame(df["smiles"])

# Converting to molecules
df_molecules["mols"] = df_molecules["smiles"].apply(Chem.MolFromSmiles)

# Morgan Fingerprint
# radius=2 = ECFP4, radius=3 = ECFP6, etc.
df_molecules["ECFP4"] = df_molecules["mols"].apply(lambda x: rdMolDescriptors.GetMorganFingerprintAsBitVect(x, 2, 1024)).apply(convert_numpy_to_list)
print(df_molecules)

#New DF with one column for each ECFP bit
df_ecfp = df_molecules['ECFP4'].apply(pd.Series)
print(df_ecfp)
