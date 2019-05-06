from rdkit.Chem import Descriptors


def calc_descriptors(df_molecules, write=False):
    # Make a copy of the molecule dataframe
    df_mols_desc = df_molecules.iloc[:, 0:2].copy()

    # Create the descriptors
    df_mols_desc["molweight"] = df_mols_desc["mols"].apply(Descriptors.ExactMolWt)
    df_mols_desc["densmorgan1"] = df_mols_desc["mols"].apply(Descriptors.FpDensityMorgan1)
    df_mols_desc["densmorgan2"] = df_mols_desc["mols"].apply(Descriptors.FpDensityMorgan2)
    df_mols_desc["densmorgan3"] = df_mols_desc["mols"].apply(Descriptors.FpDensityMorgan3)
    df_mols_desc["hatommolwt"] = df_mols_desc["mols"].apply(Descriptors.HeavyAtomMolWt)
    df_mols_desc["maxabspartcharge"] = df_mols_desc["mols"].apply(Descriptors.MaxAbsPartialCharge)
    df_mols_desc["maxpartcharge"] = df_mols_desc["mols"].apply(Descriptors.MaxPartialCharge)
    df_mols_desc["minabspc"] = df_mols_desc["mols"].apply(Descriptors.MinAbsPartialCharge)
    df_mols_desc["minpartcahrge"] = df_mols_desc["mols"].apply(Descriptors.MinPartialCharge)
    df_mols_desc["molwt"] = df_mols_desc["mols"].apply(Descriptors.MolWt)
    df_mols_desc["numrade"] = df_mols_desc["mols"].apply(Descriptors.NumRadicalElectrons)
    df_mols_desc["numval"] = df_mols_desc["mols"].apply(Descriptors.NumValenceElectrons)

    #Fill NaN with 0
    df_mols_desc = df_mols_desc.fillna(0)

    if write:
        df_mols_desc.to_csv("./dataframes/df_mols_desc.csv")

    return df_mols_desc


