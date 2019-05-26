from mlprocess import *
from params_by_label import *
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

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

# counts = y_all.sum(axis=0)
# counts.plot(kind='bar', figsize = (14,8), title="Counts of Side Effects")

# Balancing the datasets for each label
train_series_dic_bal, y_dic_bal = balance_dataset(X_train_dic, y_train_dic, out_names, random_state=seed, n_jobs=-2,
                                                  verbose=True)

# ML MODELS
# SVC
print("Base SVC without balancing:")
base_svc_report = cv_multi_report(X_train_dic, y_train, out_names, SVC(gamma="auto", random_state=seed), cv=10,
                                  n_jobs=-2, verbose=True)
ax = base_svc_report.plot.barh(y=["F1", "Recall", "Precision"])

print()
print("Base SVC with balancing:")
base_bal_svc_report = cv_multi_report(train_series_dic_bal, y_dic_bal, out_names, SVC(gamma="auto", random_state=seed),
                                      cv=10,
                                      n_jobs=-2, verbose=True)
ax2 = base_bal_svc_report.plot.barh(y=["F1", "Recall", "Precision"])
diff_bal = base_bal_svc_report - base_svc_report

# params_to_test = {"kernel": ["linear", "rbf"], "C": [0.01, 0.1, 1, 10, 100], "gamma": [0.0001, 0.001, 0.01, 0.1, 1]}
# best_svc_params_by_label = multi_label_grid_search(train_series_dic_bal, y_dic_bal, out_names, SVC(gamma="auto", random_state=seed),
# params_to_test, cv=5, scoring="f1", n_jobs=-2, verbose=True)

print()
print("Improved SVC with balancing:")
impr_bal_svc_report = cv_multi_report(train_series_dic_bal, y_dic_bal, out_names, SVC(random_state=seed),
                                      modelname="SVC", spec_params=best_svc_params_by_label, cv=10, n_jobs=-2,
                                      verbose=True)
diff_impr = impr_bal_svc_report - base_bal_svc_report
ax2 = diff_impr.plot.barh()

# RF
print()
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
ax3 = diff_bal_rf.plot.barh()

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
impr_bal_RF_report = cv_multi_report(train_series_dic_bal, y_dic_bal, out_names,
                                     RandomForestClassifier(random_state=seed), modelname="RF",
                                     spec_params=best_RF_params_by_label_grid, cv=10, n_jobs=-2, verbose=True)

diff_impr_rf = impr_bal_RF_report - base_bal_rf_report
ax2 = diff_impr_rf.plot.barh()

'''
# Test XGbBoost
print()
print("Base XGBoost:")
base_xgb = xgb.XGBClassifier(objective="binary:logistic", random_state=seed).fit(X_train, y_train)
y_pred = base_xgb.predict(X_test)
score_report(base_xgb, X_test, y_test)

eta = [0.05, 0.1, 0.2, 0.3]
min_child_weight = [1, 3, 5]
max_depth = [3, 6, 9, 12]
gamma = [0, 0.2, 0.4]
subsample = [0.1, 0.5, 1]
colsample_bytree = [0.1, 0.5, 1]
params = {"eta": eta,
          "min_child_weight": min_child_weight,
          "max_depth": max_depth,
          "gamma": gamma,
          "subsample": subsample,
          "colsample_bytree": colsample_bytree
          }
best_random_xgb = random_search(X_train, X_test, y_train, y_test,
                                xgb.XGBClassifier(objective="binary:logistic", random_state=seed),
                                grid=params, n_iter=300, cv=3, scoring="f1", n_jobs=-2, verbose=True)
# {"subsample": 1, "min_child_weight": 1, "max_depth": 12, "gamma": 0, "eta": 0.1, "colsample_bytree": 0.1}
eta = [0.01, 0.02, 0.03, 0.04]
min_child_weight = [6]
max_depth = [8]
gamma = [0.2]
subsample = [0.8]
colsample_bytree = [0.3]
params_grid = {"eta": eta,
               "min_child_weight": min_child_weight,
               "max_depth": max_depth,
               "gamma": gamma,
               "subsample": subsample,
               "colsample_bytree": colsample_bytree
               }

best_rf = grid_search(X_train, X_test, y_train, y_test,
                      xgb.XGBClassifier(objective="binary:logistic", random_state=seed), params_grid, cv=10,
                      scoring="f1", n_jobs=-2, verbose=True)
# {"colsample_bytree": 0.3, "eta": 0.05, "gamma": 0.3, "max_depth": 9, "min_child_weight": 5, "subsample": 0.8}
# {"colsample_bytree": 0.3, "eta": 0.04, "gamma": 0.2, "max_depth": 8, "min_child_weight": 6, "subsample": 0.8}
# {"colsample_bytree": 0.3, "eta": 0.03, "gamma": 0.2, "max_depth": 8, "min_child_weight": 6, "subsample": 0.8}
# {"colsample_bytree": 0.3, "eta": 0.01, "gamma": 0.2, "max_depth": 8, "min_child_weight": 6, "subsample": 0.8}
'''
