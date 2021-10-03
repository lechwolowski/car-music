import pandas as pd


contextual_rating = pd.read_excel("Data_InCarMusic.xlsx", sheet_name="ContextualRating")
contextual_rating.columns = [x.strip() for x in contextual_rating.columns.tolist()]
data_encoded = pd.get_dummies(contextual_rating)
data_encoded.to_csv("dataset.csv", sep=",", index=False, header=True)
