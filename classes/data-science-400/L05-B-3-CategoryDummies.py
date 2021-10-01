"""
# UW Data Science
# Please run code snippets one at a time to understand what is happening.
# Snippet blocks are sectioned off with a line of ####################
"""

# import package
import pandas as pd

# Download the data
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/mammographic-masses/mammographic_masses.data"
Mamm = pd.read_csv(url, header=None)
Mamm.columns = ["BI-RADS", "Age", "Shape", "Margin", "Density", "Severity"] 
#############

# The category columns are decoded, missing values are imputed, and categories
# are consolidated
Mamm.loc[ Mamm.loc[:, "Shape"] == "1", "Shape"] = "oval"
Mamm.loc[Mamm.loc[:, "Shape"] == "2", "Shape"] = "oval"
Mamm.loc[Mamm.loc[:, "Shape"] == "3", "Shape"] = "lobular"
Mamm.loc[Mamm.loc[:, "Shape"] == "4", "Shape"] = "irregular"
Mamm.loc[Mamm.loc[:, "Shape"] == "?", "Shape"] = "irregular"
Mamm.loc[Mamm.loc[:, "Margin"] == "1", "Margin"] = "circumscribed"
Mamm.loc[Mamm.loc[:, "Margin"] == "2", "Margin"] = "ill-defined"
Mamm.loc[Mamm.loc[:, "Margin"] == "3", "Margin"] = "ill-defined"
Mamm.loc[Mamm.loc[:, "Margin"] == "4", "Margin"] = "ill-defined"
Mamm.loc[Mamm.loc[:, "Margin"] == "5", "Margin"] = "spiculated"
Mamm.loc[Mamm.loc[:, "Margin"] == "?", "Margin"] = "circumscribed"
##############

# Create 3 new columns, one for each state in "Shape"
Mamm.loc[:, "oval"] = (Mamm.loc[:, "Shape"] == "oval").astype(int)
Mamm.loc[:, "lobul"] = (Mamm.loc[:, "Shape"] == "lobular").astype(int)
Mamm.loc[:, "irreg"] = (Mamm.loc[:, "Shape"] == "irregular").astype(int)
##############

# Remove obsolete column
Mamm = Mamm.drop("Shape", axis=1)
##############

# Create 3 new columns, one for each state in "Margin"
Mamm.loc[:, "ill-d"] = (Mamm.loc[:, "Margin"] == "ill-defined").astype(int)
Mamm.loc[:, "circu"] = (Mamm.loc[:, "Margin"] == "circumscribed").astype(int)
Mamm.loc[:, "spicu"] = (Mamm.loc[:, "Margin"] == "spiculated").astype(int)
##############

# Remove obsolete column
Mamm = Mamm.drop("Margin", axis=1)
##############

# Check the first rows of the data frame
Mamm.head()

# Check the data types
Mamm.dtypes
##############