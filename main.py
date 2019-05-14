import pandas as pd
import numpy as np
import create_fingerprints as cf
import create_descriptors as cd
from rdkit import Chem
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split, cross_validate
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, matthews_corrcoef
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
ecfp_df, maccs_df, atom_pairs_df, tt_df = createfingerprints(512)
df_mols_desc = createdescriptors()


#Machine learning process
#fixing the seed
seed = 6 # um número qualquer
np.random.seed(seed)

#Split into X and Y vars
X = ecfp_df.copy()
#X = pd.concat([ecfp_df, df_mols_desc], axis=1)
y = df["Hepatobiliary disorders"].copy()

estimator = SVC(gamma = 0.1)
scoring_metrics = ['roc_auc', 'accuracy', 'precision', 'recall']

cv_scores = cross_validate(estimator, X, y, scoring=scoring_metrics, cv = 10, return_train_score=False)
for k, v in cv_scores.items():
    if k != "fit_time" and k != "score_time":
        print("Métrica: %s" % k)
        print("Resultados de cada fold: %s" % v)  # resultados de cada fold
        print("Média de todas as folds: %s" % np.mean(v))
        print("Desvio padrão: %s" % np.std(v))

