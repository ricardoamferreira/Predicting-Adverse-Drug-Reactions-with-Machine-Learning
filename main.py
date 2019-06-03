from mlprocess import *
from params_by_label import *
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from pprint import pprint
import pandas as pd
import numpy as np
import xgboost as xgb

# Fixing the seed
seed = 6
np.random.seed(seed)

# Creating base df_molecules, df_y with the results vectors, and df_mols_descr with the descriptors
print("Creating Dataframes")
y_all, df_molecules = create_original_df(write=False)
df_molecules.drop("smiles", axis=1, inplace=True)
todrop = ["Product issues", "Investigations", "Social circumstances"]
y_all.drop(todrop, axis=1, inplace=True)  # No real connection with the molecule, multiple problems
out_names = y_all.columns.tolist()  # Get class labels

# Separating in a DF_mols_train and an Df_mols_test, in order to avoid data snooping and fitting the model to the test
df_mols_train, df_mols_test, y_train, y_test = train_test_split(df_molecules, y_all, test_size=0.2, random_state=seed)

# Fingerprint length
# all_df_results_svc = test_fingerprint_size(df_mols_train, all_y_train, SVC(gamma="scale", random_state=seed),
#                                            makeplots=False, write=False)
# Best result with ECFP-4 at 1125 - For now this will be used to all results

# Create X datasets with fingerprint length
X_all, _, _, _ = createfingerprints(df_molecules, length=1125)
X_train_fp, _, _, _ = createfingerprints(df_mols_train, length=1125)
X_test_fp, _, _, _ = createfingerprints(df_mols_test, length=1125)

# Selects and create descriptors dataset
df_desc = createdescriptors(df_molecules)  # Create all descriptors

# Splits in train and test
df_desc_base_train, df_desc_base_test = train_test_split(df_desc, test_size=0.2, random_state=seed)

# Creates a dictionary with key = class label and value = dataframe with fingerprint + best K descriptors for that label
X_train_dic, X_test_dic, selected_cols = create_dataframes_dic(df_desc_base_train, df_desc_base_test, X_train_fp,
                                                               X_test_fp, y_train, out_names, score_func=f_classif, k=3)
# Creates a y dictionary for all labels
y_train_dic = {name: y_train[name] for name in out_names}
modelnamesvc = {name: "SVC" for name in out_names}
modelnamerf = {name: "RF" for name in out_names}
modelnamexgb = {name: "XGB" for name in out_names}
# counts = y_all.sum(axis=0)
# counts.plot(kind='bar', figsize = (14,8), title="Counts of Side Effects")

# Balancing the datasets for each label

# print()
# print("Balancing datasets")
# train_series_dic_bal, y_dic_bal = balance_dataset(X_train_dic, y_train_dic, out_names, random_state=seed, n_jobs=-2,
#                                                verbose=True)

# ML MODELS
# SVC
print("SVC")
print("Base SVC without balancing:")
base_svc_report = cv_multi_report(X_train_dic, y_train, out_names, SVC(gamma="auto", random_state=seed), n_splits=5,
                                  n_jobs=-2, verbose=True)
# ax = base_svc_report.plot.barh(y=["F1", "Recall", "Precision"])

print()
print("Base SVC with balancing:")
base_bal_svc_report = cv_multi_report(X_train_dic, y_train, out_names, SVC(gamma="auto", random_state=seed),
                                      balancing=True, n_splits=5, n_jobs=-2, verbose=True)
diff_bal_svc = base_bal_svc_report - base_svc_report
diff_bal_svc.plot(kind="barh", y="F1")

# Searching best parameters
params_to_test = {"svc__kernel": ["rbf"], "svc__C": [0.01, 0.1, 1, 10],
                  "svc__gamma": [0.001, 0.01, 0.1, 1]}
d_params_to_test = {name: params_to_test for name in out_names}
best_svc_params_by_label = multi_label_grid_search(X_train_dic, y_train, out_names[5:10],
                                                   SVC(gamma="auto", random_state=seed), d_params_to_test,
                                                   balancing=True, n_splits=5, scoring="f1", n_jobs=-3, verbose=True,
                                                   random_state=seed)

# No changes done after this yet for balacing changes


print()
print("Improved SVC with balancing:")
impr_bal_svc_report = cv_multi_report(train_series_dic_bal, y_dic_bal, out_names, SVC(random_state=seed),
                                      modelname=modelnamesvc, spec_params=best_SVC_params_by_label, cv=10, n_jobs=-2,
                                      verbose=True)
impr_bal_svc_report = cv_multi_report(train_series_dic_bal, y_dic_bal, out_names, modelname=modelnamesvc,
                                      spec_params=best_SVC_params_by_label, random_state=seed, cv=10, n_jobs=-2,
                                      verbose=True)
diff_impr_svc = impr_bal_svc_report - base_bal_svc_report
# ax2 = diff_impr.plot.barh()

# RF
print()
print("Random Forest")
print("Base RF without balancing:")
base_rf_report = cv_multi_report(X_train_dic, y_train, out_names,
                                 RandomForestClassifier(n_estimators=100, random_state=seed), cv=10, n_jobs=-2,
                                 verbose=True)

print()
print("Base RF with balancing:")
base_bal_rf_report = cv_multi_report(train_series_dic_bal, y_dic_bal, out_names,
                                     RandomForestClassifier(n_estimators=100, random_state=seed), cv=10, n_jobs=-2,
                                     verbose=True)
diff_bal_rf = base_bal_rf_report - base_rf_report
# ax3 = diff_bal_rf.plot.barh()

# Random
n_estimators = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
max_features = ["log2", "sqrt"]
max_depth = [50, 70, 90, 110, 130, 150, 170, 190, 210, 230, 250, None]
min_samples_split = [2, 5, 10, 20]
min_samples_leaf = [1, 2, 5, 10]
bootstrap = [True, False]
rf_grid = {"n_estimators": n_estimators,
           "max_features": max_features,
           "max_depth": max_depth,
           "min_samples_split": min_samples_split,
           "min_samples_leaf": min_samples_leaf,
           "bootstrap": bootstrap}
rf_grid_label = {name: rf_grid for name in out_names}

#
# best_RF_params_by_label_random = multi_label_random_search(train_series_dic_bal, y_dic_bal, out_names,
#                                                     RandomForestClassifier(random_state=seed), rf_grid_label,
#                                                     n_iter=300,
#                                                     cv=5, scoring="f1", n_jobs=-2, verbose=True)
#
# best_RF_params_by_label_grid = multi_label_grid_search(train_series_dic_bal, y_dic_bal, out_names,
#                                                        RandomForestClassifier(random_state=seed), rf_params_to_grid,
#                                                        cv=5, scoring="f1", n_jobs=-2, verbose=True)


print()
print("Improved RF with balancing:")
impr_bal_RF_report = cv_multi_report(train_series_dic_bal, y_dic_bal, out_names, modelname=modelnamerf,
                                     spec_params=best_RF_params_by_label_grid, random_state=seed, cv=10, n_jobs=-2,
                                     verbose=True)

diff_impr_rf = impr_bal_RF_report - base_bal_rf_report
# ax2 = diff_impr_rf.plot.barh()


# XGBoost
print()
print("XGBoost")
print("Base XGBoost:")
base_xgb_report = cv_multi_report(X_train_dic, y_train, out_names,
                                  xgb.XGBClassifier(objective="binary:logistic", random_state=seed), cv=10, n_jobs=-2,
                                  verbose=True)

print()
print("Base XGBoost with balancing:")
base_bal_xgb_report = cv_multi_report(train_series_dic_bal, y_dic_bal, out_names,
                                      xgb.XGBClassifier(objective="binary:logistic", random_state=seed), cv=10,
                                      n_jobs=-2, verbose=True)
diff_bal_xgb = base_bal_xgb_report - base_xgb_report
# diff_bal_xgb.plot.barh()

# eta = [0.05, 0.1, 0.2]
# min_child_weight = [1, 3, 5]
# max_depth = [3, 5, 7, 9]
# gamma = [0, 0.1, 0.2, 0.3, 0.4]
# subsample = [0.6, 0.7, 0.8, 0.9]
# colsample_bytree = [0.6, 0.7, 0.8, 0.9]
# params = {"eta": eta,
#           "min_child_weight": min_child_weight,
#           "max_depth": max_depth,
#           "gamma": gamma,
#           "subsample": subsample,
#           "colsample_bytree": colsample_bytree
#           }
#
# xgb_grid_label = {name: params for name in out_names}
#
# best_random_xgb = multi_label_random_search(train_series_dic_bal, y_dic_bal, out_names,
#                                            xgb.XGBClassifier(objective="binary:logistic", random_state=seed),
#                                            xgb_grid_label, n_iter=300, cv=5, scoring="f1", n_jobs=-2, verbose=True)

print()
print("Improved XGB with balancing:")

impr_bal_xgb_report = cv_multi_report(train_series_dic_bal, y_dic_bal, out_names, modelname=modelnamexgb,
                                      spec_params=best_random_xgb, random_state=seed, cv=10, n_jobs=-2, verbose=True)

diff_impr_xgb = impr_bal_xgb_report - base_bal_xgb_report
# xg1 = impr_bal_xgb_report.plot.barh(y=["F1","Recall"])
# xg2 = diff_impr_xgb.plot.barh()

# Checking best model for each label
# impr_bal_RF_report; impr_bal_svc_report; impr_bal_xgb_report
f1_s = {"SVC": impr_bal_svc_report["F1"],
        "RF": impr_bal_RF_report["F1"],
        "XGB": impr_bal_xgb_report["F1"]}
all_f1_score = pd.DataFrame(data=f1_s, dtype=float)

# Creating a dictionary with Key = label, value = model name
best_model_by_label = all_f1_score.idxmax(axis=1).to_dict()
pprint(best_model_by_label)
# best_SVC_params_by_label; best_RF_params_by_label_grid; best_random_xgb

# Getting params for best model
best_model_params_by_label = {}

for label in out_names:
    if best_model_by_label[label] == "SVC":
        best_model_params_by_label[label] = best_SVC_params_by_label[label]
    elif best_model_by_label[label] == "RF":
        best_model_params_by_label[label] = best_RF_params_by_label_grid[label]
    elif best_model_by_label[label] == "XGB":
        best_model_params_by_label[label] = best_random_xgb[label]
    else:
        print(f"Error {label}")

# CV scores for best model for each label
scores_best_model = cv_multi_report(train_series_dic_bal, y_dic_bal, out_names, modelname=best_model_by_label,
                                    spec_params=best_model_params_by_label, random_state=seed, cv=10, n_jobs=-2,
                                    verbose=True)

ax = scores_best_model.sort_values(by=["F1"]).plot(kind="barh", y=["Recall", "F1"], title="Best scores by label",
                                                   xticks=[0.5, 0.6, 0.7, 0.8, 0.9, 1], legend="reverse", xlim=(0.5, 1))
for p in ax.patches: ax.annotate("{:.3f}".format(round(p.get_width(), 3)), (p.get_x() + p.get_width(), p.get_y()),
                                 xytext=(30, 0), textcoords='offset points', horizontalalignment='right')

# Test scores for each label
test_scores_best_model = test_score_multi_report(train_series_dic_bal, y_dic_bal, X_test_dic, y_test, out_names,
                                                 modelname=best_model_by_label, spec_params=best_model_params_by_label,
                                                 random_state=seed, verbose=True)

for l, df in X_test_dic.items():
    df.columns = np.arange(len(df.columns))

train_series_dic_bal["Congenital, familial and genetic disorders"]
X_test_dic["Congenital, familial and genetic disorders"]
