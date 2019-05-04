import pandas as pd
from rdkit import Chem
import create_fingerprints as cf


def create_original_df():
    global df, df_molecules
    # Create dataframe from csv
    df = pd.read_csv("./datasets/sider.csv")

    # Extract SMILES column
    df_molecules = pd.DataFrame(df["smiles"])

    # Converting to molecules
    df_molecules["mols"] = df_molecules["smiles"].apply(Chem.MolFromSmiles)
    df_molecules.to_csv("./datasets/df_molecules.csv")


def createfingerprints():
    global ecfp_df, maccs_df, atom_pairs_df

    # Morgan Fingerprint (ECFP4)
    ecfp_df = cf.create_ecfp4_fingerprint(df_molecules, False)

    # MACCS keys
    maccs_df = cf.create_maccs_fingerprint(df_molecules, False)

    # ATOM PAIRS
    atom_pairs_df = cf.create_atompairs_fingerprint(df_molecules, False)


if __name__ == "__main__":
    create_original_df()
    #createfingerprints()
    #print(df_molecules)
    #print(ecfp_df)
    #print(maccs_df)
    #print(atom_pairs_df)
