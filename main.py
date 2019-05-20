from mlprocess import *
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
# ['Hepatobiliary disorders', 'Metabolism and nutrition disorders', 'Eye disorders',
# 'Musculoskeletal and connective tissue disorders', 'Gastrointestinal disorders', 'Immune system disorders',
# 'Reproductive system and breast disorders', 'Neoplasms benign, malignant and unspecified (incl cysts and polyps)',
# 'General disorders and administration site conditions', 'Endocrine disorders', 'Surgical and medical procedures',
# 'Vascular disorders', 'Blood and lymphatic system disorders', 'Skin and subcutaneous tissue disorders',
# 'Congenital, familial and genetic disorders', 'Infections and infestations',
# 'Respiratory, thoracic and mediastinal disorders', 'Psychiatric disorders', 'Renal and urinary disorders',
# 'Pregnancy, puerperium and perinatal conditions', 'Ear and labyrinth disorders', 'Cardiac disorders',
# 'Nervous system disorders', 'Injury, poisoning and procedural complications']


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




# NOT CHECKED BELOW HERE


# Test SVC parameters


# Test RF
print()
print("Base RF:")
base_rf = RandomForestClassifier(random_state=seed).fit(X_train, y_train)
score_report(base_rf, X_test, y_test)
# F1 score: 0.66
# ROC-AUC score: 0.66
# Recall score: 0.63
# Precision score: 0.70
# Accuracy score: 0.66

print("Random Search RF")
# n_estimators = [int(x) for x in np.linspace(500, 1000, 10, dtype=int)]
n_estimators = [720, 740, 760, 780, 800, 820, 840]
max_features = ["log2"]
# max_depth = [int(x) for x in np.linspace(50, 300, 10, dtype=int)]
max_depth = [70, 80, 90, 100, 110]
min_samples_split = [9, 10, 11, 12]
min_samples_leaf = [1]
bootstrap = [True]

random_grid = {"n_estimators": n_estimators,
               "max_features": max_features,
               "max_depth": max_depth,
               "min_samples_split": min_samples_split,
               "min_samples_leaf": min_samples_leaf,
               "bootstrap": bootstrap}

best_random_rf = random_search(X_train, X_test, y_train, y_test, RandomForestClassifier(random_state=seed),
                               grid=random_grid, n_iter=300, cv=3, scoring="f1", n_jobs=-2, verbose=True)

best_rf = grid_search(X_train, X_test, y_train, y_test, RandomForestClassifier(random_state=seed), random_grid, cv=3,
                      scoring="f1", n_jobs=-2, verbose=True)
# {"bootstrap": True, "max_depth": 110, "max_features": "log2", "min_samples_leaf": 1, "min_samples_split": 10, "n_estimators": 800}

print()
print("Improved SVC Parameters")
impr_rf = RandomForestClassifier(bootstrap=True, max_depth=110, max_features="log2", min_samples_leaf=1,
                                 min_samples_split=10, n_estimators=800, random_state=seed).fit(X_train, y_train)
score_report(impr_rf, X_test, y_test)

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


### EXTRA Results

best_svc_params_by_label = {'Blood and lymphatic system disorders': {'C': 10,
                                                                     'gamma': 0.1,
                                                                     'kernel': 'rbf'},
                            'Cardiac disorders': {'C': 100,
                                                  'gamma': 0.1,
                                                  'kernel': 'rbf'},
                            'Congenital, familial and genetic disorders': {'C': 10,
                                                                           'gamma': 0.1,
                                                                           'kernel': 'rbf'},
                            'Ear and labyrinth disorders': {'C': 100,
                                                            'gamma': 0.01,
                                                            'kernel': 'rbf'},
                            'Endocrine disorders': {'C': 1,
                                                    'gamma': 0.1,
                                                    'kernel': 'rbf'},
                            'Eye disorders': {'C': 10,
                                              'gamma': 0.1,
                                              'kernel': 'rbf'},
                            'Gastrointestinal disorders': {'C': 10,
                                                           'gamma': 0.1,
                                                           'kernel': 'rbf'},
                            'General disorders and administration site conditions': {'C': 10,
                                                                                     'gamma': 0.1,
                                                                                     'kernel': 'rbf'},
                            'Hepatobiliary disorders': {'C': 1,
                                                        'gamma': 0.1,
                                                        'kernel': 'rbf'},
                            'Immune system disorders': {'C': 10,
                                                        'gamma': 0.1,
                                                        'kernel': 'rbf'},
                            'Infections and infestations': {'C': 10,
                                                            'gamma': 0.1,
                                                            'kernel': 'rbf'},
                            'Injury, poisoning and procedural complications': {'C': 10,
                                                                               'gamma': 0.1,
                                                                               'kernel': 'rbf'},
                            'Metabolism and nutrition disorders': {'C': 10,
                                                                   'gamma': 0.1,
                                                                   'kernel': 'rbf'},
                            'Musculoskeletal and connective tissue disorders': {'C': 10,
                                                                                'gamma': 0.1,
                                                                                'kernel': 'rbf'},
                            'Neoplasms benign, malignant and unspecified (incl cysts and polyps)': {'C': 10,
                                                                                                    'gamma': 0.1,
                                                                                                    'kernel': 'rbf'},
                            'Nervous system disorders': {'C': 100,
                                                         'gamma': 0.1,
                                                         'kernel': 'rbf'},
                            'Pregnancy, puerperium and perinatal conditions': {'C': 10,
                                                                               'gamma': 0.1,
                                                                               'kernel': 'rbf'},
                            'Psychiatric disorders': {'C': 10,
                                                      'gamma': 0.1,
                                                      'kernel': 'rbf'},
                            'Renal and urinary disorders': {'C': 100,
                                                            'gamma': 0.1,
                                                            'kernel': 'rbf'},
                            'Reproductive system and breast disorders': {'C': 10,
                                                                         'gamma': 0.1,
                                                                         'kernel': 'rbf'},
                            'Respiratory, thoracic and mediastinal disorders': {'C': 10,
                                                                                'gamma': 0.1,
                                                                                'kernel': 'rbf'},
                            'Skin and subcutaneous tissue disorders': {'C': 10,
                                                                       'gamma': 0.1,
                                                                       'kernel': 'rbf'},
                            'Surgical and medical procedures': {'C': 0.1,
                                                                'gamma': 0.0001,
                                                                'kernel': 'linear'},
                            'Vascular disorders': {'C': 10,
                                                   'gamma': 0.1,
                                                   'kernel': 'rbf'}}
