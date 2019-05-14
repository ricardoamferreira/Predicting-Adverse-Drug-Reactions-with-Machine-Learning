import pandas as pd
import create_fingerprints as cf
import create_descriptors as cd
from rdkit import Chem


def create_original_df(write=False):
    # Create dataframe from csv
    df = pd.read_csv("./datasets/sider.csv")

    # Extract SMILES column
    df_molecules = pd.DataFrame(df["smiles"])

    # Converting to molecules
    df_molecules["mols"] = df_molecules["smiles"].apply(Chem.MolFromSmiles)

    # Write to csv
    if write:
        df_molecules.to_csv("./dataframes/df_molecules.csv")

    return df_molecules


def createfingerprints(length):
    global ecfp_df, maccs_df, atom_pairs_df, tt_df

    # Morgan Fingerprint (ECFP4)
    ecfp_df = cf.create_ecfp4_fingerprint(df_molecules, length, False)

    # MACCS keys (always 167)
    maccs_df = cf.create_maccs_fingerprint(df_molecules, False)

    # ATOM PAIRS
    atom_pairs_df = cf.create_atompairs_fingerprint(df_molecules, length, False)

    # Topological torsion
    tt_df = cf.create_topological_torsion_fingerprint(df_molecules, length, False)

    return ecfp_df, maccs_df, atom_pairs_df, tt_df

def createdescriptors():

    # Descriptors
    df_mols_desc = cd.calc_descriptors(df_molecules, True)

    return df_mols_desc


df_molecules = create_original_df(write=False)
#ecfp_df, maccs_df, atom_pairs_df, tt_df = createfingerprints(512)
df_mols_desc = createdescriptors()
