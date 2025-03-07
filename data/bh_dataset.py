"""Script to clean and run Buchwald Hartwig regression.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from rxnfp.transformer_fingerprints import (
    RXNBERTFingerprintGenerator, get_default_model_and_tokenizer
)
from sklearn.metrics import r2_score


def train_and_predict_rf(X_train: np.array, 
                         y_train: np.array):
    """Train and predict using RF regressor
    """
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    return rf


def split_train_regressors(data_path: str,
                           save_path: str):
    """Split and train the regressors on the dataset and generate predictions.
    """

    # read data
    df = pd.read_csv(data_path)

    # load rxnfp generator 
    model, tokenizer = get_default_model_and_tokenizer()

    rxnfp_generator = RXNBERTFingerprintGenerator(model, tokenizer)

    # create rxnfps
    df["fp"] = df["rxn"].apply(rxnfp_generator.convert)

    # split data
    df_95, df_5 = train_test_split(df, 
                                    test_size=0.05, 
                                    random_state=33)

    df_rest, df_30 = train_test_split(df_95,
                                      test_size=0.31, 
                                      random_state=33)
    
    # Train low-fidelity regressor on 10% data
    X_train_5 = np.stack(df_5["fp"].values)
    y_train_5 = df_5["yield"].values
    rf_10 = train_and_predict_rf(X_train_5, y_train_5)

    # Train medium-fidelity regressor on 30% data
    X_train_30 = np.stack(df_30["fp"].values)
    y_train_30 = df_30["yield"].values
    rf_30 = train_and_predict_rf(X_train_30, y_train_30)

    # predict with both to create LF and MF data
    lf_predictions = rf_10.predict(np.stack(df_rest["fp"].values))
    mf_predictions = rf_30.predict(np.stack(df_rest["fp"].values))

    # create new columns, save df
    df_rest["LF"] = lf_predictions
    df_rest["MF"] = mf_predictions

    df_rest.rename(columns={"yield": "HF"}, inplace=True)

    df_rest = df_rest[["rxn", "fp", "LF", "MF", "HF"]]

    # save new dataset
    df_rest.to_csv(save_path,
                   index=False)


DATA_PATH = "data/raw/Merged_BH_Reaction_Data.csv"
SAVE_PATH = "data/clean/BH_dataset.csv"

if __name__ == "__main__":

    split_train_regressors(DATA_PATH,
                           SAVE_PATH)
    

    bh = pd.read_csv(SAVE_PATH)

    # Plot LF and MF vs HF
    plt.figure(figsize=(10, 6))
    plt.scatter(bh['HF'], bh['LF'], label='LF vs HF', alpha=0.5)
    plt.scatter(bh['HF'], bh['MF'], label='MedF vs HF', alpha=0.5)
    plt.plot([bh['HF'].min(), bh['HF'].max()], [bh['HF'].min(), bh['HF'].max()], 'k--', label='x = y', c="r")
    plt.xlabel('HF yield')
    plt.ylabel('LF / MedF yield')
    plt.legend()
    plt.title('Low fidelity (LF) and Medium fidelity (MedF) vs High fidelity (HF)')

    # Compute R2 for LF vs HF
    r2_lf = r2_score(bh['HF'], bh['LF'])
    plt.text(0.05, 0.95, f'R² for LF vs HF: {r2_lf:.2f}', transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))

    # Compute R2 for MF vs HF
    r2_mf = r2_score(bh['HF'], bh['MF'])
    plt.text(0.05, 0.90, f'R² for MedF vs HF: {r2_mf:.2f}', transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    plt.savefig("plots/regression/BH_fidelities_regression.png")