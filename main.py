import pandas as pd
import create_fingerprints as cf
import create_descriptors as cd
from rdkit import Chem


def create_original_df(write=False):
    global df, df_molecules
    # Create dataframe from csv
    df = pd.read_csv("./datasets/sider.csv")

    # Extract SMILES column
    df_molecules = pd.DataFrame(df["smiles"])

    # Converting to molecules
    df_molecules["mols"] = df_molecules["smiles"].apply(Chem.MolFromSmiles)

    # Write to csv
    if write:
        df_molecules.to_csv("./dataframes/df_molecules.csv")


def createfingerprints():
    global ecfp_df, maccs_df, atom_pairs_df, tt_df

    # Morgan Fingerprint (ECFP4)
    ecfp_df = cf.create_ecfp4_fingerprint(df_molecules, False)

    # MACCS keys
    maccs_df = cf.create_maccs_fingerprint(df_molecules, False)

    # ATOM PAIRS
    atom_pairs_df = cf.create_atompairs_fingerprint(df_molecules, False)

    # Topological torsion
    tt_df = cf.create_topological_torsion_fingerprint(df_molecules, False)


def createdescriptors():
    global df_mols_desc

    #Descriptors
    df_mols_desc = cd.calc_descriptors(df_molecules, False)



if __name__ == "__main__":
    create_original_df(write=False)
    createfingerprints()
    createdescriptors()
