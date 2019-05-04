import pandas as pd
import numpy as np
from aux_functions import to_numpyarray_to_list
from rdkit.Chem import rdMolDescriptors


def get_morgan_with_try(molecule):
    try:
        # radius=2 = ECFP4, radius=3 = ECFP6, etc.
        desc = rdMolDescriptors.GetMorganFingerprintAsBitVect(molecule, 2, 1024)
    except Exception as e:
        print(e)
        print('error ' + str(molecule))
        desc = np.nan
    return desc


def get_maccs_with_try(molecule):
    try:
        maccs = rdMolDescriptors.GetMACCSKeysFingerprint(molecule)
    except Exception as e:
        print(e)
        print("error" + str(molecule))
        maccs = np.nan
    return maccs


def get_atompairs_with_try(molecule):
    try:
        atompairs = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(molecule)
    except Exception as e:
        print(e)
        print("error" + str(molecule))
        atompairs = np.nan
    return atompairs


def create_ecfp4_fingerprint(df_molecules, write=False):
    # Morgan Fingerprint (ECFP4)
    df_molecules["ECFP4"] = df_molecules["mols"].apply(get_morgan_with_try).apply(to_numpyarray_to_list)

    # New DF with one column for each ECFP bit
    ecfp_df = df_molecules['ECFP4'].apply(pd.Series)
    ecfp_df = ecfp_df.rename(columns=lambda x: 'ECFP4_' + str(x + 1))

    # Write to csv
    if write:
        ecfp_df.to_csv("./datasets/ecfp4.csv")

    return ecfp_df


def create_maccs_fingerprint(df_molecules, write=False):
    # MACCS keys
    df_molecules["MACCS"] = df_molecules["mols"].apply(get_maccs_with_try).apply(to_numpyarray_to_list)

    # New DF with one column for each MACCS key
    maccs_df = df_molecules['MACCS'].apply(pd.Series)
    maccs_df = maccs_df.rename(columns=lambda x: 'MACCS_' + str(x + 1))

    # Write to csv
    if write:
        maccs_df.to_csv("./datasets/maccs.csv")

    return maccs_df


def create_atompairs_fingerprint(df_molecules, write=False):
    # ATOM PAIRS
    df_molecules["ATOMPAIRS"] = df_molecules["mols"].apply(get_atompairs_with_try).apply(to_numpyarray_to_list)

    # New DF with one column for each ATOM PAIRS key
    atom_pairs_df = df_molecules['ATOMPAIRS'].apply(pd.Series)
    atom_pairs_df = atom_pairs_df.rename(columns=lambda x: 'ATOMPAIR_' + str(x + 1))

    # Write to csv
    if write:
        atom_pairs_df.to_csv("./datasets/atom_pairs.csv")

    return atom_pairs_df
