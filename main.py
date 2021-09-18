import pandas as pd

dataset = {
    "ContextualRating": pd.read_excel("Data_InCarMusic.xlsx", sheet_name="ContextualRating"),
    "Context Factor": pd.read_excel("Data_InCarMusic.xlsx", sheet_name="Context Factor"),
    "Music Track": pd.read_excel("Data_InCarMusic.xlsx", sheet_name="Music Track"),
    "Music Category": pd.read_excel("Data_InCarMusic.xlsx", sheet_name="Music Category"),
}

print(dataset)