import pandas as pd
import numpy as np
import create_fingerprints as cf
import create_descriptors as cd
from rdkit import Chem
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split, cross_validate
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
import matplotlib.pyplot as plt


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
scoring_metrics = ["roc_auc", "precision", "recall", "accuracy"]

sizes = np.linspace(100, 2048, 20, dtype=int)

results_rocauc = np.zeros([4, len(sizes)])
results_precision = np.zeros([4, len(sizes)])
results_recall = np.zeros([4, len(sizes)])
results_accuracy = np.zeros([4, len(sizes)])


c = 0
r = 0
#Size testing
for s in sizes:
    print(f"Doing size {int(s)}")
    fingerprints = createfingerprints(int(s))
    r = 0
    for fp in fingerprints:
        X = fp.copy()
        y = df["Hepatobiliary disorders"].copy()

        cv_scores = cross_validate(SVC(gamma = "scale"), X, y, cv=10, scoring=scoring_metrics, return_train_score= False)


        for k, v in cv_scores.items():
            if k == "test_roc_auc":
                results_rocauc[r, c] = v.mean()
            if k == "test_precision":
                results_precision[r, c] = v.mean()
            if k == "test_recall":
                results_recall[r, c] = v.mean()
            if k == "test_accuracy":
                results_accuracy[r, c] = v.mean()
        r += 1
    c += 1


df_results_rocauc_size_SVC = pd.DataFrame(results_rocauc, columns=sizes)
df_results_precision_size_SVC = pd.DataFrame(results_precision, columns=sizes)
df_results_recall_size_SVC = pd.DataFrame(results_recall, columns=sizes)
df_results_accuracy_size_SVC = pd.DataFrame(results_accuracy, columns=sizes)

df_results_rocauc_size_SVC.to_csv("./results/df_results_rocauc_size_SVC")
df_results_precision_size_SVC.to_csv("./results/df_results_precision_size_SVC")
df_results_recall_size_SVC.to_csv("./results/df_results_recall_size_SVC")
df_results_accuracy_size_SVC.to_csv("./results/df_results_accuracy_size_SVC")



plt.clf()
fig = plt.figure(figsize=(10,10))
fp_names = ["ecfp","maccs","atom pairs","tt"]
for i in range(len(fingerprints)):
    plt.plot(sizes, results_precision[i,:], "-")
plt.title("SVM, ROC-AUC vs fingerprint length",fontsize=25)
plt.ylabel("ROC-AUC", fontsize = 20)
plt.xlabel("ROC-AUC", fontsize = 20)
plt.legend(fp_names, fontsize = 15)
plt.ylim([0,1])
plt.show()








