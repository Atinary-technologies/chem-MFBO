"""Utils to clean and encode datasets."""
import pandas as pd
import torch
from rdkit import Chem

def inchi_to_smiles(inchi: str) -> str:
    """Transform Inchi into SMILES"""
    try:
        smiles = Chem.MolToSmiles(Chem.MolFromInchi(inchi))
        return smiles
    except:
        return False


def has_carbon(smiles: str) -> bool:
    """Check if molecule contains the alkyl motif"""
    mol = Chem.MolFromSmiles(smiles)
    carb = mol.HasSubstructMatch(Chem.MolFromSmarts("[#6][#6]"))

    return carb


def polarizability_to_input(path: str) -> pd.DataFrame:
    """Transform polarizability dataset to correct format.
    """

    columns = [
        'ID',
        'Compound',
        'Formula',
        'Charge',
        'Multiplicity',
        'CAS_no',
        'ChemSpider_ID',
        'PubChem_ID',
        'Rotatable_Bonds',
        'StdInChI',
        'InChIkey',
        'Spe1',
        'Exp_Dipole',
        'Error_Dipole',
        'Ref_Dipole',
        'HF_6_311G_Dipole',
        'Sep2',
        'Exp_pol',
        'Error_Polarizability',
        'Ref_Polarizability',
        'HF_6_311G_pol',
        'Sep3',
        'Filename',
    ]

    df = pd.read_csv(
        path, delimiter='|', comment='#', skiprows=9, header=None, names=columns
    )

    # drop molecules that don't have polarizability values
    df = df.dropna(subset="Exp_pol")

    # transform SMILES into Inchi
    df["smiles"] = df["StdInChI"].apply(inchi_to_smiles)

    df = df[df['smiles'] != False]

    # take only those who have an alkyl chain (organic molecules)
    df = df[df['smiles'].apply(has_carbon)]

    # take only SMILES, HF_6_311G_pol and DR columns
    df = df[["smiles", "Exp_pol", "HF_6_311G_pol"]]

    # rename columns
    df = df.rename(columns={"Exp_pol": "HF", "HF_6_311G_pol": "LF"})

    return df


def bandgap_to_input(path: str) -> pd.DataFrame:
    """Transform bandgap data to input
    """
    df = pd.read_csv(path)

    # exclude extra redundant info in the last columns
    df = df.iloc[:, 0:-3]

    # Create a pivot table to merge the rows
    merged_df = df.pivot_table(index='Composition', columns='expt_data', values='Bandgap', aggfunc='first').reset_index()

    # Rename the columns
    merged_df.columns = ['Composition', 'LF', 'HF']

    # Select unique rows from df while removing 'Bandgap', 'theory_data', and 'expt_data'
    unique_df = df.drop(columns=['Bandgap', 'theory_data', 'expt_data']).drop_duplicates()

    # Merge the unique_df with merged_df based on the 'Composition' key
    combined_df = pd.merge(merged_df, unique_df, on='Composition', how='inner')

    # Remove the 'Composition' column
    combined_df = combined_df.drop(columns=['Composition'])

    # Reorder columns to put 'HF' and 'LF' as the last two columns
    cols = [col for col in combined_df.columns if col not in ['HF', 'LF']] + ['HF', 'LF']
    
    combined_df = combined_df[cols]

    return combined_df
