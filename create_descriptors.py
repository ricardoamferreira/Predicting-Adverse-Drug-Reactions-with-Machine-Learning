from rdkit.Chem import Descriptors, Lipinski


def calc_descriptors(df_molecules, write=False):
    # Make a copy of the molecule dataframe
    df_mols_desc = df_molecules.copy()

    # Create the descriptors (9)
    df_mols_desc["molweight"] = df_mols_desc["mols"].apply(Descriptors.ExactMolWt)
    df_mols_desc["hatommolwt"] = df_mols_desc["mols"].apply(Descriptors.HeavyAtomMolWt)
    df_mols_desc["maxabspartcharge"] = df_mols_desc["mols"].apply(Descriptors.MaxAbsPartialCharge)
    df_mols_desc["maxpartcharge"] = df_mols_desc["mols"].apply(Descriptors.MaxPartialCharge)
    df_mols_desc["minabspc"] = df_mols_desc["mols"].apply(Descriptors.MinAbsPartialCharge)
    df_mols_desc["minpartcharge"] = df_mols_desc["mols"].apply(Descriptors.MinPartialCharge)
    df_mols_desc["molwt"] = df_mols_desc["mols"].apply(Descriptors.MolWt)
    df_mols_desc["numrade"] = df_mols_desc["mols"].apply(Descriptors.NumRadicalElectrons)
    df_mols_desc["numval"] = df_mols_desc["mols"].apply(Descriptors.NumValenceElectrons)

    #Lipinski (18)
    df_mols_desc["fracsp33"] = df_mols_desc["mols"].apply(Lipinski.FractionCSP3)
    df_mols_desc["heavyatomcount"] = df_mols_desc["mols"].apply(Lipinski.HeavyAtomCount)
    df_mols_desc["nhohcount"] = df_mols_desc["mols"].apply(Lipinski.NHOHCount)
    df_mols_desc["nocount"] = df_mols_desc["mols"].apply(Lipinski.NOCount)
    df_mols_desc["aliphcarbocycles"] = df_mols_desc["mols"].apply(Lipinski.NumAliphaticCarbocycles)
    df_mols_desc["aliphhetcycles"] = df_mols_desc["mols"].apply(Lipinski.NumAliphaticHeterocycles)
    df_mols_desc["aliphrings"] = df_mols_desc["mols"].apply(Lipinski.NumAliphaticRings)
    df_mols_desc["arocarbocycles"] = df_mols_desc["mols"].apply(Lipinski.NumAromaticCarbocycles)
    df_mols_desc["arohetcycles"] = df_mols_desc["mols"].apply(Lipinski.NumAromaticHeterocycles)
    df_mols_desc["arorings"] = df_mols_desc["mols"].apply(Lipinski.NumAromaticRings)
    df_mols_desc["numhacceptors"] = df_mols_desc["mols"].apply(Lipinski.NumHAcceptors)
    df_mols_desc["numhdonors"] = df_mols_desc["mols"].apply(Lipinski.NumHDonors)
    df_mols_desc["numhatoms"] = df_mols_desc["mols"].apply(Lipinski.NumHeteroatoms)
    df_mols_desc["numrotbonds"] = df_mols_desc["mols"].apply(Lipinski.NumRotatableBonds)
    df_mols_desc["numsatcarbcycles"] = df_mols_desc["mols"].apply(Lipinski.NumSaturatedCarbocycles)
    df_mols_desc["numsathetcycles"] = df_mols_desc["mols"].apply(Lipinski.NumSaturatedHeterocycles)
    df_mols_desc["numsatrings"] = df_mols_desc["mols"].apply(Lipinski.NumSaturatedRings)
    df_mols_desc["ringcount"] = df_mols_desc["mols"].apply(Lipinski.RingCount)

    #Drop SMILES and MOLS
    df_mols_desc.drop("mols", inplace=True, axis=1)


    #Fill NaN with 0
    df_mols_desc = df_mols_desc.fillna(0)

    if write:
        df_mols_desc.to_csv("./dataframes/df_mols_desc.csv")

    return df_mols_desc


