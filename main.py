# Misc
from sklearn.model_selection import train_test_split
from pprint import pprint

# Functions
from mlprocess import *
from params_by_label import *

# Fixing the seed
seed = 6
np.random.seed(seed)

""" Careful when importing saved dataframes with label names as indexes -> index = True """

# Creating base df_molecules, df_y with the results vectors, and df_mols_descr with the descriptors
print("Creating Dataframes")
y_all, df_molecules = create_original_df(write_s=False)
df_molecules.drop("smiles", axis=1, inplace=True)
todrop = ["Product issues", "Investigations", "Social circumstances"]
y_all.drop(todrop, axis=1, inplace=True)  # No real connection with the molecule, multiple problems
out_names = y_all.columns.tolist()  # Get class labels

# Separating in a DF_mols_train and an Df_mols_test, in order to avoid data snooping and fitting the model to the test
df_mols_train, df_mols_test, y_train, y_test = train_test_split(df_molecules, y_all, test_size=0.2, random_state=seed)

# Fingerprint length
# all_df_results_svc = test_fingerprint_size(df_mols_train, y_train, SVC(gamma="scale", random_state=seed), makeplots=True, write=True)
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
modelnamevot = {name: "VotingClassifier" for name in out_names}

""" Analysis """
d = {"Positives": y_all.sum(axis=0), "Negatives": 1427 - y_all.sum(axis=0)}
countsm = pd.DataFrame(data=d)
df_perc = countsm / 1427
df_3filt = df_perc.loc[["Hepatobiliary disorders", "Gastrointestinal disorders",
                        "Neoplasms benign, malignant and unspecified (incl cysts and polyps)"]]
df_3filt.to_csv("./results/df_3filt.csv", float_format='%.3f')
countsm.plot(kind='bar', figsize=(14, 8), title="Adverse Drug Reactions Counts", ylim=(0, 1500), stacked=True)
df_perc.plot(kind="bar")

"""ML PROCESS"""
# SVC
print("SVC")
print("Base SVC without balancing:")
base_svc_report = cv_multi_report(X_train_dic, y_train, out_names, SVC(gamma="auto", random_state=seed), n_splits=5,
                                  n_jobs=-2, verbose=True)

# base_svc_report.to_csv("./results/base_svc_report.csv")


print()
print("Base SVC with balancing:")
base_bal_svc_report = cv_multi_report(X_train_dic, y_train, out_names, SVC(gamma="auto", random_state=seed),
                                      balancing=True, n_splits=5, n_jobs=-2, verbose=True, random_state=seed)

# base_bal_svc_report.to_csv("./results/base_bal_svc_report.csv")
# diff_bal_svc = base_bal_svc_report - base_svc_report
# diff_bal_svc.to_csv("./results/diff_bal_svc.csv", float_format='%.3f')
# diff_bal_svc.plot(kind="barh", y=["Average Precision", "F1 Micro", "F1 Macro", "F1 Binary"],
#                   title="Oversampling Changes")


# Searching best parameters
# params_to_test = {"svc__kernel": ["rbf"], "svc__C": [0.01, 0.1, 1, 10],
#                   "svc__gamma": [0.001, 0.01, 0.1, 1]}
# d_params_to_test = {name: params_to_test for name in out_names}
# best_SVC_params_by_label = multi_label_grid_search(X_train_dic, y_train, out_names[15:],
#                                                    SVC(gamma="auto", random_state=seed), d_params_to_test,
#                                                    balancing=True, n_splits=5, scoring="f1_micro", n_jobs=-2,
#                                                    verbose=True, random_state=seed)

print()
print("Improved SVC with balancing:")
impr_bal_svc_report = cv_multi_report(X_train_dic, y_train, out_names, modelname=modelnamesvc,
                                      spec_params=best_SVC_params_by_label, balancing=True, n_splits=5, n_jobs=-2,
                                      verbose=True, random_state=seed)

# impr_bal_svc_report.to_csv("./results/impr_bal_svc_report.csv", float_format='%.3f')
# impr_bal_svc_report = pd.read_csv("./results/impr_bal_svc_report.csv", index_col=0)
# diff_impr_svc = impr_bal_svc_report - base_bal_svc_report
# diff_impr_svc.to_csv("./results/diff_impr_svc.csv", float_format='%.3f')
# diff_impr_svc = pd.read_csv("./results/diff_impr_svc.csv", index_col=0)
# ax2 = diff_impr.plot.barh()

# RF
print()
print("Random Forest")
print("Base RF without balancing:")
base_rf_report = cv_multi_report(X_train_dic, y_train, out_names,
                                 RandomForestClassifier(n_estimators=100, random_state=seed), n_splits=5, n_jobs=-2,
                                 verbose=True)
# base_rf_report.to_csv("./results/base_rf_report.csv", float_format='%.3f')


print()
print("Base RF with balancing:")
base_bal_rf_report = cv_multi_report(X_train_dic, y_train, out_names,
                                     RandomForestClassifier(n_estimators=100, random_state=seed), balancing=True,
                                     n_splits=5, n_jobs=-2, verbose=True, random_state=seed)
# base_bal_rf_report.to_csv("./results/base_bal_rf_report.csv", float_format='%.3f')
# diff_bal_rf = base_bal_rf_report - base_rf_report
# ax3 = diff_bal_rf.plot.barh()

# Random
# n_estimators = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
# max_features = ["log2", "sqrt"]
# max_depth = [50, 90, 130, 170, 210, 250]
# min_samples_split = [2, 5, 10]
# min_samples_leaf = [1, 2]
# bootstrap = [True, False]
# rf_grid = {"randomforestclassifier__n_estimators": n_estimators,
#            "randomforestclassifier__max_features": max_features,
#            "randomforestclassifier__max_depth": max_depth,
#            "randomforestclassifier__min_samples_split": min_samples_split,
#            "randomforestclassifier__min_samples_leaf": min_samples_leaf,
#            "randomforestclassifier__bootstrap": bootstrap}
# rf_grid_label = {name: rf_grid for name in out_names}
# best_RF_params_by_label = multi_label_random_search(X_train_dic, y_train, out_names[20:],
#                                                     RandomForestClassifier(random_state=seed), rf_grid_label,
#                                                     balancing=True, n_splits=3, scoring="f1_micro", n_jobs=-2,
#                                                     verbose=True, random_state=seed, n_iter=150)

print()
print("Improved RF with balancing:")
impr_bal_rf_report = cv_multi_report(X_train_dic, y_train, out_names, modelname=modelnamerf,
                                     spec_params=best_RF_params_by_label, balancing=True, n_splits=5, n_jobs=-2,
                                     verbose=True, random_state=seed)
impr_bal_rf_report.to_csv("./results/impr_bal_rf_report.csv", float_format='%.3f')
# diff_impr_rf = impr_bal_rf_report - base_bal_rf_report
# ax2 = diff_impr_rf.plot.barh()


# XGBoost
print()
print("XGBoost")
print("Base XGBoost:")
base_xgb_report = cv_multi_report(X_train_dic, y_train, out_names,
                                  xgb.XGBClassifier(objective="binary:logistic", random_state=seed), n_splits=5,
                                  n_jobs=-2, verbose=True)
base_xgb_report.to_csv("./results/base_xgb_report.csv", float_format='%.3f')

print()
print("Base XGBoost with balancing:")
base_bal_xgb_report = cv_multi_report(X_train_dic, y_train, out_names,
                                      xgb.XGBClassifier(objective="binary:logistic", random_state=seed), balancing=True,
                                      n_splits=5, n_jobs=-2, verbose=True, random_state=seed)
base_bal_xgb_report.to_csv("./results/base_bal_xgb_report.csv", float_format='%.3f')
diff_bal_xgb = base_bal_xgb_report - base_xgb_report
# diff_bal_xgb.plot.barh()


# eta = [0.05, 0.1, 0.2]
# min_child_weight = [1, 3]
# max_depth = [5, 7, 9]
# gamma = [0, 0.1, 0.2, 0.3, 0.4]
# subsample = [0.6, 0.7, 0.8, 0.9]
# colsample_bytree = [0.6, 0.7, 0.8, 0.9]
# xgb_grid = {"xgbclassifier__eta": eta,
#             "xgbclassifier__min_child_weight": min_child_weight,
#             "xgbclassifier__max_depth": max_depth,
#             "xgbclassifier__gamma": gamma,
#             "xgbclassifier__subsample": subsample,
#             "xgbclassifier__colsample_bytree": colsample_bytree
#             }
# xgb_grid_label = {name: xgb_grid for name in out_names}
# best_xgb_params_by_label = multi_label_random_search(X_train_dic, y_train, out_names[20:],
#                                                      xgb.XGBClassifier(objective="binary:logistic", random_state=seed),
#                                                      xgb_grid_label, balancing=True, n_splits=3, scoring="f1_micro",
#                                                      n_jobs=-2, verbose=True, random_state=seed, n_iter=150)

print()
print("Improved XGB with balancing:")

impr_bal_xgb_report = cv_multi_report(X_train_dic, y_train, out_names, modelname=modelnamexgb,
                                      spec_params=best_xgb_params_by_label, balancing=True, n_splits=5, n_jobs=-2,
                                      verbose=True, random_state=seed)
# impr_bal_xgb_report.to_csv("./results/impr_bal_xgb_report.csv", float_format='%.3f')
# impr_bal_xgb_report.sort_values(by=["Average Precision"], ascending=False, inplace=True)
# impr_bal_xgb_report.to_csv("./results/impr_bal_xgb_report.csv")
# diff_impr_xgb = impr_bal_xgb_report - base_bal_xgb_report
##xg1 = impr_bal_xgb_report.plot.barh(y=["F1","Recall"])
# # xg2 = diff_impr_xgb.plot.barh()
#


print()
print("Voting classifier: ")

impr_bal_xgb_report = cv_multi_report(X_train_dic, y_train, out_names, modelname=modelnamexgb,
                                      spec_params=best_xgb_params_by_label, balancing=True, n_splits=5, n_jobs=-2,
                                      verbose=True, random_state=seed)

# Checking best model for each label
# impr_bal_RF_report; impr_bal_svc_report; impr_bal_xgb_report


f1_mi_s = {"SVC": impr_bal_svc_report["F1 Micro"],
           "RF": impr_bal_rf_report["F1 Micro"],
           "XGB": impr_bal_xgb_report["F1 Micro"]}
all_f1_mi_score = pd.DataFrame(data=f1_mi_s, dtype=float)

f1_ma_s = {"SVC": impr_bal_svc_report["F1 Macro"],
           "RF": impr_bal_rf_report["F1 Macro"],
           "XGB": impr_bal_xgb_report["F1 Macro"]}
all_f1_ma_score = pd.DataFrame(data=f1_ma_s, dtype=float)

f1_b_s = {"SVC": impr_bal_svc_report["F1 Binary"],
          "RF": impr_bal_rf_report["F1 Binary"],
          "XGB": impr_bal_xgb_report["F1 Binary"]}
all_b_ma_score = pd.DataFrame(data=f1_b_s, dtype=float)

roc_s = {"SVC": impr_bal_svc_report["ROC_AUC"],
         "RF": impr_bal_rf_report["ROC_AUC"],
         "XGB": impr_bal_xgb_report["ROC_AUC"]}
all_roc_score = pd.DataFrame(data=roc_s, dtype=float)

rec_s = {"SVC": impr_bal_svc_report["Recall"],
         "RF": impr_bal_rf_report["Recall"],
         "XGB": impr_bal_xgb_report["Recall"]}
all_rec_score = pd.DataFrame(data=rec_s, dtype=float)

prec_s = {"SVC": impr_bal_svc_report["Precision"],
          "RF": impr_bal_rf_report["Precision"],
          "XGB": impr_bal_xgb_report["Precision"]}
all_prec_score = pd.DataFrame(data=prec_s, dtype=float)

av_prec = {"SVC": impr_bal_svc_report["Average Precision"],
           "RF": impr_bal_rf_report["Average Precision"],
           "XGB": impr_bal_xgb_report["Average Precision"]}
all_av_prec = pd.DataFrame(data=av_prec, dtype=float)

# Creating a dictionary with Key = label, value = model name
# best_model_by_label = all_f1_score.idxmax(axis=1).to_dict()
# pprint(best_model_by_label)
# best_SVC_params_by_label; best_rf_params_by_label_grid; best_xgb_params_by_label

# Getting params for best model
best_model_params_by_label = {}

for label in out_names:
    if best_model_by_label[label] == "SVC":
        best_model_params_by_label[label] = best_SVC_params_by_label[label]
    elif best_model_by_label[label] == "RF":
        best_model_params_by_label[label] = best_RF_params_by_label[label]
    elif best_model_by_label[label] == "XGB":
        best_model_params_by_label[label] = best_xgb_params_by_label[label]
    else:
        print(f"Error {label}")

# # CV scores for best model for each label
scores_best_model = cv_multi_report(X_train_dic, y_train, out_names, modelname=best_model_by_label,
                                    spec_params=best_model_params_by_label, balancing=True, n_splits=5, n_jobs=-2,
                                    verbose=True, random_state=seed)
scores_best_model.to_csv("./results/scores_best_model.csv", float_format='%.3f')

# Best model cv score graph
ax = scores_best_model.plot(kind="barh",
                            y=["Average Precision", "F1 Macro", "F1 Micro"],
                            title="Best scores by label",
                            xticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                                    0.9, 1],
                            legend="reverse", xlim=(0, 1))
for p in ax.patches: ax.annotate("{:.3f}".format(round(p.get_width(), 3)), (p.get_x() + p.get_width(), p.get_y()),
                                 xytext=(30, 0), textcoords='offset points', horizontalalignment='right')

# Test score voting
scores_voting = cv_multi_report(X_train_dic, y_train, out_names, modelname=modelnamevot,
                                spec_params=(
                                    best_SVC_params_by_label, best_RF_params_by_label, best_xgb_params_by_label),
                                balancing=True, n_splits=5, n_jobs=-2, verbose=True, random_state=seed)

# Test scores for each label
test_scores_best_model = test_score_multi_report(X_train_dic, y_train, X_test_dic, y_test, out_names,
                                                 modelname=best_model_by_label, spec_params=best_model_params_by_label,
                                                 random_state=seed, verbose=True, balancing=True, n_jobs=-2, plot=True)

# test_scores_best_model.sort_values(by=["Average Prec-Rec"], ascending=False, inplace=True)
# test_scores_best_model.to_csv("./results/test_scores_best_model.csv")
ax = test_scores_best_model.sort_values(by=["Average Prec-Rec"]).plot(kind="barh",
                                                                      y=["Average Prec-Rec", "ROC_AUC"],
                                                                      title="Best scores by label",
                                                                      xticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
                                                                              0.9, 1],
                                                                      legend="reverse", xlim=(0, 1))
for p in ax.patches: ax.annotate("{:.3f}".format(round(p.get_width(), 3)), (p.get_x() + p.get_width(), p.get_y()),
                                 xytext=(30, 0), textcoords='offset points', horizontalalignment='right')

"""OFFSIDES DATASET"""
# mod_off = create_offside_df(out_names=out_names, write=False)
mod_off = pd.read_csv("./datasets/offside_socs_modified.csv")
df = pd.read_csv("./datasets/sider.csv")
todrop = ["Product issues", "Investigations", "Social circumstances"]
df.drop(todrop, axis=1, inplace=True)
dups = set(mod_off.smiles).intersection(df.smiles)
len(dups)  # 716 Duplicates with different information

# 1332 Rows in Total
df_y_2 = mod_off.drop("smiles", axis=1)
d2 = {"Positives": df_y_2.sum(axis=0), "Negatives": 1332 - df_y_2.sum(axis=0)}
counts = pd.DataFrame(data=d2)
counts.plot(kind='bar', figsize=(14, 8), title="Adverse Drug Reactions Counts", ylim=(0, 1400), stacked=True)

# Merging datasets
# doff = mod_off.loc[mod_off["smiles"].isin(dups), :].copy()
# dsid = df.loc[df["smiles"].isin(dups), :].copy()
#
# doff.sort_values(by=["smiles"], inplace=True)
# dsid.sort_values(by=["smiles"], inplace=True)
#
# dfd = {"smiles": list(dups)}
# df_dups = pd.DataFrame(data=dfd)
#
# for name in out_names:
#     df_dups[name] = 0
#
# for index, row in tqdm(doff.iterrows()):
#     for name in out_names:
#         if row[name] == 1:
#             df_dups.loc[df_dups["smiles"] == row["smiles"], name] = row[name]
#
# for index, row in tqdm(dsid.iterrows()):
#     for name in out_names:
#         if row[name] == 1:
#             df_dups.loc[df_dups["smiles"] == row["smiles"], name] = row[name]
#
# df_wo_ofs = df.loc[~df["smiles"].isin(dups), :].copy()  # (711, 28)
# df_wo_sid = mod_off.loc[~mod_off["smiles"].isin(dups), :].copy()  # (711, 28)
#
# df_all = pd.concat([df_wo_ofs, df_wo_sid, df_dups], axis=0, sort=False)
# df_all.shape  # (2043, 25)
# df_all.to_csv("./dataframes/df_all.csv", index=False)

df_all = pd.read_csv("./dataframes/df_all.csv")  # (2043, 25)

# New counts (SIDER + OFFSIDES)
df_all_y = df_all.drop("smiles", axis=1)
da2 = {"Positives": df_all_y.sum(axis=0), "Negatives": 2043 - df_all_y.sum(axis=0)}
counts = pd.DataFrame(data=da2)
counts.plot(kind='bar', figsize=(14, 8), title="Adverse Drug Reactions Counts (SIDER + OFFSIDES)", ylim=(0, 2100),
            stacked=True)

# df_perc is percentage in sider
perc_da2 = counts / 2043

diff_from_sider = perc_da2 - df_perc

df_off_y, df_off_mols = create_original_df(usedf=True, file=df_all, write_s=False, write_off=False)
df_off_mols.drop("smiles", axis=1, inplace=True)

df_off_mols_train, df_off_mols_test, y_off_train, y_off_test = train_test_split(df_off_mols, df_off_y, test_size=0.2,
                                                                                random_state=seed)

# Create X datasets with fingerprint length
X_off_all, _, _, _ = createfingerprints(df_off_mols, length=1125)
X_off_train_fp, _, _, _ = createfingerprints(df_off_mols_train, length=1125)
X_off_test_fp, _, _, _ = createfingerprints(df_off_mols_test, length=1125)

# Selects and create descriptors dataset
df_off_desc = createdescriptors(df_off_mols)  # Create all descriptors

# Splits in train and test
df_off_desc_base_train, df_off_desc_base_test = train_test_split(df_off_desc, test_size=0.2, random_state=seed)

# Creates a dictionary with key = class label and value = dataframe with fingerprint + best K descriptors for that label
X_off_train_dic, X_off_test_dic, selected_off_cols = create_dataframes_dic(df_off_desc_base_train,
                                                                           df_off_desc_base_test, X_off_train_fp,
                                                                           X_off_test_fp, y_off_train, out_names,
                                                                           score_func=f_classif, k=3)

test_scores_sioff = test_score_multi_report(X_off_train_dic, y_off_train, X_off_test_dic, y_off_test, out_names,
                                            modelname=best_model_by_label, spec_params=best_model_params_by_label,
                                            random_state=seed, verbose=True, balancing=True, n_jobs=-3, plot=True)
test_scores_sioff.sort_values(by=["Average Prec-Rec"], ascending=False, inplace=True)
test_scores_sioff.to_csv("./results/test_scores_sioff.csv")
test_scores_sioff = pd.read_csv("./results/test_scores_sioff.csv", index_col=0)

# Differences after joining offsides dataset
diff_offsides = test_scores_sioff - test_scores_best_model
diff_offsides.sort_values(by=["Average Prec-Rec"], ascending=False, inplace=True)
diff_offsides.to_csv("./results/diff_offsides.csv")
