import pandas as pd
import numpy as np
from aux_functions import to_numpyarray_to_list
from rdkit.Chem import rdMolDescriptors


def get_morgan(molecule):
    try:
        # radius=2 = ECFP4, radius=3 = ECFP6, etc.
        desc = rdMolDescriptors.GetMorganFingerprintAsBitVect(molecule, 2, 1024)
    except Exception as e:
        print(e)
        print('error ' + str(molecule))
        desc = np.nan
    return desc


def get_maccs(molecule):
    try:
        maccs = rdMolDescriptors.GetMACCSKeysFingerprint(molecule)
    except Exception as e:
        print(e)
        print("error" + str(molecule))
        maccs = np.nan
    return maccs


def get_atompairs(molecule):
    try:
        atompairs = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(molecule)
    except Exception as e:
        print(e)
        print("error" + str(molecule))
        atompairs = np.nan
    return atompairs


def get_topological_torsion(molecule):
    try:
        tt = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(molecule)
    except Exception as e:
        print(e)
        print("error" + str(molecule))
        tt = np.nan
    return tt


def create_ecfp4_fingerprint(df_molecules, write=False):
    # Morgan Fingerprint (ECFP4)
    df_molecules["ECFP4"] = df_molecules["mols"].apply(get_morgan).apply(to_numpyarray_to_list)

    # New DF with one column for each ECFP bit
    ecfp_df = df_molecules['ECFP4'].apply(pd.Series)
    ecfp_df = ecfp_df.rename(columns=lambda x: 'ECFP4_' + str(x + 1))

    # Write to csv
    if write:
        ecfp_df.to_csv("./dataframes/ecfp4.csv")

    return ecfp_df


def create_maccs_fingerprint(df_molecules, write=False):
    # MACCS keys
    df_molecules["MACCS"] = df_molecules["mols"].apply(get_maccs).apply(to_numpyarray_to_list)

    # New DF with one column for each MACCS key
    maccs_df = df_molecules['MACCS'].apply(pd.Series)
    maccs_df = maccs_df.rename(columns=lambda x: 'MACCS_' + str(x + 1))

    # Write to csv
    if write:
        maccs_df.to_csv("./dataframes/maccs.csv")

    return maccs_df


def create_atompairs_fingerprint(df_molecules, write=False):
    # ATOM PAIRS
    df_molecules["ATOMPAIRS"] = df_molecules["mols"].apply(get_atompairs).apply(to_numpyarray_to_list)

    # New DF with one column for each ATOM PAIRS key
    atom_pairs_df = df_molecules['ATOMPAIRS'].apply(pd.Series)
    atom_pairs_df = atom_pairs_df.rename(columns=lambda x: 'ATOMPAIR_' + str(x + 1))

    # Write to csv
    if write:
        atom_pairs_df.to_csv("./dataframes/atom_pairs.csv")

    return atom_pairs_df


def create_topological_torsion_fingerprint(df_molecules, write=False):
    # Topological Torsion
    df_molecules["TT"] = df_molecules["mols"].apply(get_atompairs).apply(to_numpyarray_to_list)

    # New DF with one column for each Topological torsion key
    tt_df = df_molecules['TT'].apply(pd.Series)
    tt_df = tt_df.rename(columns=lambda x: 'TT' + str(x + 1))

    # Write to csv
    if write:
        tt_df.to_csv("./dataframes/topological_torsion.csv")

    return tt_df
