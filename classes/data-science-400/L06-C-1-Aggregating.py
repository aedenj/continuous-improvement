"""
# UW Data Science
# Please run code snippets one at a time to understand what is happening.
# Snippet blocks are sectioned off with a line of ####################
"""

import pandas as pd

# you can have pandas convert the date directly during the import
# Be patient, this step will take some processing time (30+seconds depending on internet connection)
pm2_file = pd.read_csv("https://library.startlearninglabs.uw.edu/DATASCI400/Datasets/BeijingPM2_IOT.csv",
                       parse_dates=['TimeStamp'],
                       infer_datetime_format=True) 

print(pm2_file.head())
# What kind of data is present?
#########################

# chained subset for temp, precip, humidity
# use copy function to return a new dataframe instead of referring back to original
pm2_df = pm2_file[(pm2_file['Attribute'] == 'precipitation') | 
        (pm2_file['Attribute'] == 'TEMP') | 
        (pm2_file['Attribute'] == 'HUMI')].copy()
# notice the size of pm2_df

print(pm2_df.head())
# What kind of data is present?
#########################

# set time as an index so we can easily group data by date
# calling the function on the object in format of object.function(parameter)
pm2_df = pm2_df.set_index(['TimeStamp'])
# Why is Timestamp set off with quote marks?

# view measurement in 2010
print(pm2_df['2010'])
# how many rows show up?
#########################

# view specific month and year
print(pm2_df['2010-12'])
# how many rows show up?
#########################

# pull out year and month for each time stamp
print(pm2_df.index.year)
print(pm2_df.index.month)
# what years show up? what months show up?
#########################

# count number of observations per year
print(pm2_df.groupby(pm2_df.index.year).count())
# how many observations are there in 2010?
#########################
