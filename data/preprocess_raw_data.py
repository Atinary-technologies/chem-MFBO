"""Preprocess each individual raw dataset to get the correct format:
Features/SMILES,HF,LF
"""
import os
import pandas as pd
from utils import polarizability_to_input, bandgap_to_input


TARGET_FILES = ["Bandgap_data_withfeatures_CLEANED_2022-08-26.csv", 
                "freesolv.csv", 
                "polarizability.csv", 
                "converted_data_raw.csv"]


if __name__ == "__main__":

    # open raw directory
    raw_dir = os.path.join(os.path.dirname(__file__), "raw")
    files = os.listdir(raw_dir)
    
    # assert that each file is contained in the directory
    for target_file in TARGET_FILES:
        assert target_file in files, f"{target_file} is not in the raw directory"

    # loop through each file and transform it to the corresponding benchmark structure
    for file in files:

        if file == "freesolv.csv":

            df = pd.read_csv(os.path.join(raw_dir, file))

            df = df[["smiles","expt","calc"]]

            df = df.rename(columns={"expt": "HF", "calc": "LF"})

            # we want to find the optimum, but the energy is negative, so we invert it
            df["HF"] = df["HF"] * -1
            df["LF"] = df["LF"] * -1

            name = "freesolv"

        elif file == "polarizability.csv":
            
            df = polarizability_to_input(os.path.join(raw_dir, file))
            
            name = "polarizability"
        
        elif file == "converted_data_raw.csv":
            
            df = pd.read_csv(os.path.join(raw_dir, file))
            
            f_names = list(df.columns[1:15])

            df = df[f_names + ["gcmc_y", "henry_y"]]

            df = df.rename(columns={"gcmc_y": "HF", "henry_y": "LF"})

            name = "cofs"

        elif file == "Bandgap_data_withfeatures_CLEANED_2022-08-26.csv":
            
            df = bandgap_to_input(os.path.join(raw_dir, file))

            # remove 0 values
            df = df[df["HF"] != 0]
            
            df = df.sample(1000, random_state=33)

            name = "bandgap"
        
        # save df in the clean folder
        clean_dir = os.path.join(os.path.dirname(__file__), "clean")
        os.makedirs(clean_dir, exist_ok=True)
        df.to_csv(os.path.join(clean_dir, f"{name}.csv"), index=False)

    print("Finished dataset preprocessing")

