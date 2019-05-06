import pandas as pd
import create_fingerprints as cf
from rdkit import Chem

def create_original_df(write=False):
    global df, df_molecules
    # Create dataframe from csv
    df = pd.read_csv("./datasets/sider.csv")

    # Extract SMILES column
    df_molecules = pd.DataFrame(df["smiles"])

    # Converting to molecules
    df_molecules["mols"] = df_molecules["smiles"].apply(Chem.MolFromSmiles)

    #Write to csv
    if write:
        df_molecules.to_csv("./dataframes/df_molecules.csv")


def createfingerprints():
    global ecfp_df, maccs_df, atom_pairs_df, tt_df

    # Morgan Fingerprint (ECFP4)
    ecfp_df = cf.create_ecfp4_fingerprint(df_molecules, True)

    # MACCS keys
    maccs_df = cf.create_maccs_fingerprint(df_molecules, True)

    # ATOM PAIRS
    atom_pairs_df = cf.create_atompairs_fingerprint(df_molecules, True)

    # Topological torsion
    tt_df = cf.create_topological_torsion_fingerprint(df_molecules, True)


if __name__ == "__main__":
    create_original_df(write=True)
    createfingerprints()
    print(df_molecules)
    print(ecfp_df)
    print(maccs_df)
    print(atom_pairs_df)
    print(tt_df)
