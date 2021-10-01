"""
UW Data Science
Please run code snippets one at a time to understand what is happening.
Snippet blocks are sectioned off with a line of ####################
"""

# for this scrape, we'll use the pandas package to convert csv to a dataframe
import pandas as pd

# let's start with a URL that you've worked with in past tutorials
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"

# show URL and discuss csv format

# Let's pull down the csv info directly into a pandas dataframe
adult_df = pd.read_csv(url, header=None)
print(adult_df.head())
#########################

# we can see that we successfully pulled the information into a dataframe
# but lets' add some column headers to make it easier to work with this data 
# downstream

# here we make a list of column names
adult_columns = ["age", "workclass", "fnlwgt", "education", "education-num", 
                 "marital-status", "occupation", "relationship", "race", "sex", 
                 "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]

# and now we can apply that list of names to our dataframe
adult_df.columns = adult_columns

print(adult_df.head())

#########################