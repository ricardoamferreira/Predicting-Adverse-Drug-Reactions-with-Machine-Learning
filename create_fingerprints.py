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
