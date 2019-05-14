import pandas as pd
import numpy as np
import create_fingerprints as cf
import create_descriptors as cd
from rdkit import Chem
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split, cross_validate
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif


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

    return df, df_molecules


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
    df_mols_desc = cd.calc_descriptors(df_molecules, False)

    return df_mols_desc


df, df_molecules = create_original_df(write=False)

df_mols_desc = createdescriptors()


#Machine learning process
#fixing the seed
seed = 6 # um número qualquer
np.random.seed(seed)


#Split into X and Y vars
#X = ecfp_df.copy()
y = df["Hepatobiliary disorders"].copy()
scoring_metrics = ["f1"]

sizes = np.linspace(100, 2048, 20)

results = np.zeros([4, len(sizes)])

c = 0
r = 0

for s in sizes:
    print(f"Doing size {int(s)}")
    fingerprints = createfingerprints(int(s))
    r = 0
    for fp in fingerprints:
        X = fp.copy()
        y = df["Hepatobiliary disorders"].copy()

        cv_scores = cross_validate(SVC(gamma = "auto"), X, y, cv=10, scoring=scoring_metrics, return_train_score= False)


        for k, v in cv_scores.items():
            if k != "fit_time" and k != "score_time":
                results[r, c] = v.mean()
        r += 1
    c += 1

print(results)








