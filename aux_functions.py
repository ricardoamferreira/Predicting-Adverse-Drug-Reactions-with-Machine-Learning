import numpy as np
from rdkit import DataStructs
from rdkit.Chem import rdMolDescriptors


def getMorganWithTry(molecule):
    try:
        # radius=2 = ECFP4, radius=3 = ECFP6, etc.
        desc = rdMolDescriptors.GetMorganFingerprintAsBitVect(molecule, 2, 1024)
    except Exception as e:
        print(e)
        print('error ' + str(molecule))
        desc = np.nan
    return desc


def tonumpyarraytolist(desc):
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(desc, arr)
    return arr.tolist()
