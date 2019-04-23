import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.nan)
from rdkit import Chem, DataStructs
from rdkit.Chem import rdMolDescriptors


def get_first_smile(smile):
    return smile.split(';')[0]


def convert_numpy_to_list(np_array):
    return np_array.tolist()


def get_ecfp(compound_df, radius, output_file):

    # radius=2 = ECFP4, radius=3 = ECFP6, etc.

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

    # Drop compounds that don't have SMILES (this step may not be necessary)
    df = compound_df.loc[compound_df['SMILES'].notnull(), :]
    # Process rows where there is more than one SMILE (this step may not be necessary)
    df['SMILES'] = df['SMILES'].apply(get_first_smile)
    # Generate ECFP4 fingerprint
    df['ECFP4'] = df['SMILES'].apply(_getMorganWithTry).apply(convert_numpy_to_list)
    # Convert fingerprint column into multiple columns (one for each bit)
    ecfp_df = df['ECFP4'].apply(pd.Series)
    ecfp_df = ecfp_df.rename(columns=lambda x: 'ECFP4_' + str(x+1))
    df = pd.concat([df, ecfp_df], axis=1).drop(['ECFP4', 'SMILES'], axis=1)

    # Save calculated fingerprints to file:
    df.to_csv(output_file, index=False)

    return df


if __name__ == '__main__':
    df = pd.read_csv('Drug_info_release_modified2.csv')
    get_ecfp(compound_df=df, radius=2, output_file='ecfp4.csv')