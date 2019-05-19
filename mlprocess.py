import pandas as pd
import numpy as np
import create_fingerprints as cf
import create_descriptors as cd
from rdkit import Chem
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, cross_validate, RandomizedSearchCV
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFECV
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score, f1_score, \
    accuracy_score, roc_auc_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import xgboost as xgb
from tqdm import tqdm


def create_original_df(write=False):
    # Create dataframe from csv
    df = pd.read_csv("./datasets/sider.csv")

    # Extract SMILES column
    df_molecules = pd.DataFrame(df["smiles"])

    # Converting to molecules
    df_molecules["mols"] = df_molecules["smiles"].apply(Chem.MolFromSmiles)

    # Droping mols and smiles
    df_y = df.drop("smiles", axis=1)

    # Write to csv
    if write:
        df_molecules.to_csv("./dataframes/df_molecules.csv")
        df_y.to_csv("./dataframes/df_y.csv")

    return df_y, df_molecules


def createfingerprints(df_mols, length):
    # Morgan Fingerprint (ECFP4)
    ecfp_df = cf.create_ecfp4_fingerprint(df_mols, length, False)

    # MACCS keys (always 167)
    maccs_df = cf.create_maccs_fingerprint(df_mols, False)

    # ATOM PAIRS
    atom_pairs_df = cf.create_atompairs_fingerprint(df_mols, length, False)

    # Topological torsion
    tt_df = cf.create_topological_torsion_fingerprint(df_mols, length, False)

    return ecfp_df, maccs_df, atom_pairs_df, tt_df


def createdescriptors(df_molecules):
    # Descriptors
    df_mols_desc = cd.calc_descriptors(df_molecules, False)

    return df_mols_desc


def test_fingerprint_size(df_mols, df_y, model, colname="Hepatobiliary disorders", num_sizes_to_test=20, min_size=100,
                          max_size=2048, cv=10, makeplots=False, write=False):
    # Fingerprint length type and selection
    # Scoring metrics to use
    scoring_metrics = ["roc_auc", "precision", "recall", "accuracy", "f1"]
    sizes = np.linspace(min_size, max_size, num_sizes_to_test, dtype=int)

    # Create results dataframes for each metric
    results_f1 = np.zeros([4, len(sizes)])
    results_rocauc = np.zeros([4, len(sizes)])
    results_precision = np.zeros([4, len(sizes)])
    results_recall = np.zeros([4, len(sizes)])
    results_accuracy = np.zeros([4, len(sizes)])

    # Get test sizes
    c = 0
    # Size testing using SVC with scale gamma (1 / (n_features * X.var()))
    for s in tqdm(sizes):
        # Create fingerprint with size S
        fingerprints = createfingerprints(df_mols, int(s))
        r = 0
        for fp in fingerprints:
            X = fp.copy()
            # Using "Hepatobiliary disorders" as an results example since its balanced
            y = df_y[colname].copy()
            # 10-fold cross validation
            cv_scores = cross_validate(model, X, y, cv=cv, scoring=scoring_metrics, return_train_score=False, n_jobs=-1)

            for k, v in cv_scores.items():
                if k == "test_roc_auc":
                    results_rocauc[r, c] = v.mean()
                if k == "test_precision":
                    results_precision[r, c] = v.mean()
                if k == "test_recall":
                    results_recall[r, c] = v.mean()
                if k == "test_accuracy":
                    results_accuracy[r, c] = v.mean()
                if k == "test_f1":
                    results_f1[r, c] = v.mean()
            r += 1
        c += 1

    all_results = (results_rocauc, results_precision, results_recall, results_accuracy, results_f1)

    # Create dataframe for results
    df_results_rocauc_size_SVC = pd.DataFrame(results_rocauc, columns=sizes)
    df_results_precision_size_SVC = pd.DataFrame(results_precision, columns=sizes)
    df_results_recall_size_SVC = pd.DataFrame(results_recall, columns=sizes)
    df_results_accuracy_size_SVC = pd.DataFrame(results_accuracy, columns=sizes)
    df_results_f1_size_SVC = pd.DataFrame(results_f1, columns=sizes)

    all_df_results = (
        df_results_rocauc_size_SVC, df_results_precision_size_SVC, df_results_recall_size_SVC,
        df_results_accuracy_size_SVC,
        df_results_f1_size_SVC)

    # Save to file
    if write:
        df_results_rocauc_size_SVC.to_csv("./results/df_results_rocauc_size_SVC.csv")
        df_results_precision_size_SVC.to_csv("./results/df_results_precision_size_SVC.csv")
        df_results_recall_size_SVC.to_csv("./results/df_results_recall_size_SVC.csv")
        df_results_accuracy_size_SVC.to_csv("./results/df_results_accuracy_size_SVC.csv")
        df_results_f1_size_SVC.to_csv("./results/df_results_f1_size_SVC.csv")

    if makeplots:
        fp_names = ["ECFP-4", "MACCS", "Atom Pairs", "Topological Torsion"]
        m = 0
        for d in all_results:
            fig = plt.figure(figsize=(10, 10))
            for i in range(len(fingerprints)):
                plt.plot(sizes, d[i, :], "-")
            plt.title(f"SVC, {scoring_metrics[m]} vs fingerprint length", fontsize=25)
            plt.ylabel(f"{scoring_metrics[m]}", fontsize=20)
            plt.xlabel("Fingerprint Length", fontsize=20)
            plt.legend(fp_names, fontsize=15)
            plt.ylim([0, 1])
            plt.show()
            m += 1

    return all_df_results


def select_best_descriptors_multi(df_desc, y_all, out_names=[], score_func=f_classif, k=1):
    # Select k highest scoring feature from X to every y and return new df with only the selected ones
    if not out_names:
        print("Column names necessary")
        return None
    selected = []
    for n in out_names:
        skb = SelectKBest(score_func=score_func, k=k).fit(df_desc, y_all[n])
        n_sel_bol = skb.get_support()
        sel = df_desc.loc[:, n_sel_bol].columns.to_list()
        for s in sel:
            if s not in selected:
                selected.append(s)
    return selected


def select_best_descriptors(X, y, score_func=f_classif, k=2):
    # Select k highest scoring feature from X to y with a score function, f_classif by default
    skb = SelectKBest(score_func=score_func, k=k).fit(X, y)
    n_sel_bol = skb.get_support()
    sel = X.loc[:, n_sel_bol].columns.to_list()
    assert sel
    return sel


def create_dataframes_dic(df_desc_base_train, df_desc_base_test, X_train_fp, X_test_fp, y_train, out_names,
                          score_func=f_classif, k=3):
    # Create 3 dictionaries, one with the train dataframes, one with the test dataframes and one with the selected
    # features for each label

    # Initialize dictonaries
    train_series_dic = {name: None for name in out_names}
    test_series_dic = {name: None for name in out_names}
    selected_name = {name: None for name in out_names}

    # For each of the tasks build the train and test dataframe with the selected descriptors
    for name in out_names:
        # Select best descriptors for the task
        sel_col = select_best_descriptors(df_desc_base_train, y_train[name], score_func=score_func, k=k)
        selected_name[name] = sel_col  # Keep track of selected columns
        df_desc_train = df_desc_base_train.loc[:, sel_col].copy()  # Get train dataframe with only selected columns
        df_desc_test = df_desc_base_test.loc[:, sel_col].copy()  # Get test dataframe with only selected columns
        X_train = pd.concat([X_train_fp, df_desc_train], axis=1)
        X_test = pd.concat([X_test_fp, df_desc_test], axis=1)
        # Add to the dictionary
        train_series_dic[name] = X_train
        test_series_dic[name] = X_test

    # Return the dictionaries
    return train_series_dic, test_series_dic, selected_name


def cv_multi_report(X_train_dic, y_train, out_names, model, cv=10, n_jobs=-1, verbose=False):
    # Creates a scores report dataframe for each classification label with cv
    # Initizalize the dataframe
    report = pd.DataFrame(index=["F1", "ROC_AUC", "Recall", "Precision", "Accuracy"], columns=out_names)
    scoring_metrics = ("f1", "roc_auc", "recall", "precision", "accuracy")

    # For each label
    for name in out_names:
        if verbose:
            print(f"Scores for {name}")
        # Calculate the score for the current label using the respective dataframe
        scores = cv_report(model, X_train_dic[name], y_train[name], cv=cv, scoring_metrics=scoring_metrics,
                           n_jobs=n_jobs, verbose=verbose)
        report.loc["F1", name] = round(float(scores["f1_score"]), 3)
        report.loc["ROC_AUC", name] = round(float(scores["auc_score"]), 3)
        report.loc["Recall", name] = round(float(scores["rec_score"]), 3)
        report.loc["Precision", name] = round(float(scores["prec_score"]), 3)
        report.loc["Accuracy", name] = round(float(scores["acc_score"]), 3)
    return report


def grid_search(X_train, X_test, y_train, y_test, model, params_to_test, cv=10, scoring="f1", n_jobs=-1, verbose=False):
    # Define grid search
    grid_search = GridSearchCV(model, params_to_test, cv=cv, n_jobs=n_jobs, verbose=verbose, scoring=scoring)

    # Fit X and y to test parameters
    grid_search.fit(X_train, y_train)
    means = grid_search.cv_results_["mean_test_score"]
    stds = grid_search.cv_results_["std_test_score"]

    if verbose:
        # Print scores
        print()
        print("Score for development set:")
        for mean, std, params in zip(means, stds, grid_search.cv_results_["params"]):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 1.96, params))
        print()

        # Print best parameters
        print()
        print("Best parameters set found:")
        print(grid_search.best_params_)
        print()

        # Detailed Classification report
        print()
        print("Detailed classification report:")
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, grid_search.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()
        print("Confusion Matrix as")
        print("""
        TN FP
        FN TP
        """)
        print(confusion_matrix(y_true, y_pred))
    # Save best estimator
    best_estimator = grid_search.best_estimator_
    best_params = grid_search.best_params_
    # And return it
    return best_params, best_estimator


def random_search(X_train, X_test, y_train, y_test, model, grid, n_iter=100, cv=10, scoring="f1", n_jobs=-1,
                  verbose=False):
    # Define random search
    rs = RandomizedSearchCV(model, grid, n_iter=n_iter, cv=cv, scoring=scoring, n_jobs=n_jobs, verbose=verbose)

    # Fit parameters
    rs.fit(X_train, y_train)

    # Print scores
    print()
    print("Score for development set:")
    means = rs.cv_results_["mean_test_score"]
    stds = rs.cv_results_["std_test_score"]
    for mean, std, params in zip(means, stds, rs.cv_results_["params"]):
        print("%0.3f (+/-%0.03f) for %r" % (mean, std * 1.96, params))
    print()

    # Print best parameters
    print()
    print("Best parameters set found:")
    print(rs.best_params_)
    print()

    # Detailed Classification report
    print()
    print("Detailed classification report:")
    print("The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.")
    print()
    y_true, y_pred = y_test, rs.predict(X_test)
    print(classification_report(y_true, y_pred))
    print()
    """
    print("Confusion matrix as:")
    print(
           TN FP
           FN TP
           )
    print(confusion_matrix(y_true, y_pred))
    print()
    """
    # Save best estimator
    best_estimator = rs.best_estimator_
    # And return it
    return best_estimator


def score_report(estimator, X_test, y_test):
    # Predicting value
    y_true, y_pred = y_test, estimator.predict(X_test)

    # Detailed Classification report
    print()
    print("The scores are computed on the full evaluation set")
    print("These are not used to train or optimize the model")
    print()
    print("Detailed classification report:")
    print(classification_report(y_true, y_pred))
    print()
    """
    print("Confusion matrix as:")
    print(
           #TN FP
           #FN TP
           )
    print(confusion_matrix(y_true, y_pred))
    print()
    """
    # Individual metrics
    f1 = f1_score(y_true, y_pred, average="macro")
    auc = roc_auc_score(y_true, y_pred, average="macro")
    rec = recall_score(y_true, y_pred, average="macro")
    prec = precision_score(y_true, y_pred, average="macro")
    acc = accuracy_score(y_true, y_pred)

    print("Individual metrics:")
    print(f"F1 score: {f1:.2f}")
    print(f"ROC-AUC score: {auc:.2f}")
    print(f"Recall score: {rec:.2f}")
    print(f"Precision score: {prec:.2f}")
    print(f"Accuracy score: {acc:.2f}")
    print()


def cv_report(estimator, X_train, y_train, cv=10, scoring_metrics=("f1", "roc_auc", "recall", "precision", "accuracy"),
              n_jobs=-1, verbose=False):
    # Cross validation
    scores = cross_validate(estimator, X_train, y_train, scoring=scoring_metrics, cv=cv, n_jobs=n_jobs, verbose=verbose,
                            return_train_score=False)

    # Means
    f1_s = np.mean(scores["test_f1"])
    auc_s = np.mean(scores["test_roc_auc"])
    rec_s = np.mean(scores["test_recall"])
    prec_s = np.mean(scores["test_precision"])
    acc_s = np.mean(scores["test_accuracy"])

    # STD
    f1_std = np.std(scores["test_f1"])
    auc_std = np.std(scores["test_roc_auc"])
    rec_std = np.std(scores["test_recall"])
    prec_std = np.std(scores["test_precision"])
    acc_std = np.std(scores["test_accuracy"])

    if verbose:
        print()
        print("Individual metrics")
        print(f"F1 Score: Mean: {f1_s:.3f} (Std: {f1_std:.3f})")
        print(f"ROC-AUC score: Mean: {auc_s:.3f} (Std: {auc_std:.3f})")
        print(f"Recall score: Mean: {rec_s:.3f} (Std: {rec_std:.3f})")
        print(f"Precision score: Mean: {prec_s:.3f} (Std: {prec_std:.3f})")
        print(f"Accuracy score: Mean: {acc_s:.3f} (Std: {acc_std:.3f})")
        print()

    return {"f1_score": f1_s, "f1_std": f1_std, "auc_score": auc_s, "auc_std": auc_std, "rec_score": rec_s,
            "rec_std": rec_std, "prec_score": prec_s, "prec_std": prec_std, "acc_score": acc_s, "acc_std": acc_std}
