"""
# UW Data Science
# Please run code snippets one at a time to understand what is happening.
# Snippet blocks are sectioned off with a line of ####################
"""
""" Table Manipulations"""

# import required modules
import pandas as pd
import numpy as np

###  create tables using numpy and pandas ###

# set random seed 
np.random.seed(42)

# employee sales table using numpy randint
emp_sales = np.random.randint(100,3000, size=(3,4)).astype("float")

# display the 3 x 4 table.
print(emp_sales)
##########################

# convert to pandas dataframe. A dataframe is a table.
sales_df = pd.DataFrame(np.random.randint(100,3000,size=(3, 4)), columns=list('ABCD')).astype("float")

# display the 3 x 4 table
print(sales_df.head())
#########################

# create employee info data frame from list of dictionaries
info_dict = [{"id":10115, "first_name":"Bob", "last_name":"Edwards", "comm_pct":5.6},
                 {"id":10117, "first_name":"Helen", "last_name":"Sanchez", "comm_pct":7.8},
                 {"id":19928, "first_name":"Mohammed", "last_name":"Mamali", "comm_pct":5.6}
                 ]
# This list contains 4 attributes for each employee
#########################

# Create a dataframe from the list
info_df = pd.DataFrame(info_dict)

# display the 3 x 4 table
print(info_df)
# What order are the columns in?
#########################

"""Insert a column """

# add in a column of employee id's from other table
sales_df['id'] = info_df['id']
# use the name of the column to create a new column
# if the column name does not already exist, it is created.

# display the new 3 x 5 table
print(sales_df)

# or add new column from a list of id's
# create the list of id's
emp_id_list = [10115, 10117, 19928]

#assign the list to the column named 'id'
sales_df['id'] = emp_id_list
print(sales_df)
# the output should still be the same, we used the name id numbers
#########################

""" Update column names """

# change column names using dictionary mapping 
# create the dictionary
new_col_names = {"A":"Q1_sales", "B":"Q2_sales", "C":"Q3_sales", "D":"Q4_sales"}

# assign the new names to the columns with rename function
sales_df.rename(columns=new_col_names, inplace=True)

# write a statement to display the 3 x 5 table of sales_df:

#########################

""" Merge tables """

# Join two tables based on common "id" column
print(info_df)
print(sales_df)

# use the merge function
combo_df = info_df.merge(sales_df, how="inner", on="id")

# How many columns are there in the new table? Check Variable explorer.
print(combo_df)

########################

""" Set index on a dataframe  """
# the index is like a header column used to refer to each row.
combo_df.index= info_df['id']

# display the combo_df table:

# Do we still need the id column?

# drop leftover id column (1 = by column, 0 = by row)
combo_df.drop('id', axis=1, inplace=True)
# How many columns are there in combo_df? Check the Variable explorer.
print(combo_df)
# Is the table easier to read now?

#######################

""" Accessing information in a dataframe """

# loc works on index labels
print(combo_df.loc[10115])
print(combo_df.loc[10115, ['first_name','last_name']])

# iloc works on index position which starts counting at 0
print(combo_df.iloc[0])

#######################

""" Insert a row in a dataframe """

# add new row to info_df
combo_df.loc[19917] = [4.6, "Astrid", "Moonbeam", 553,1912,198,2055]
# How many rows are there in combo_df?
print(combo_df)

########################

""" Summing a row """

# recall create a new column with a new name of a column
# create a new column containing total sales by summing each row
combo_df['ttl_sales'] = combo_df['Q1_sales'] + combo_df['Q2_sales'] + combo_df['Q3_sales'] + combo_df['Q4_sales']
print(combo_df)

# add ttl_sales column using a subset
combo_df['ttl_sales2'] = combo_df[['Q1_sales', 'Q2_sales', 'Q3_sales', 'Q4_sales']].sum()
print(combo_df)
# Whoops! Need to tell it to how to sum.
# axis=1 by column, 0 by row
combo_df['ttl_sales2'] = combo_df[['Q1_sales', 'Q2_sales', 'Q3_sales', 'Q4_sales']].sum(axis=1)
print(combo_df)

######################

""" Multiplying columns """
# multiply ttl_sales by comm_pct to find emp_ttl_comm
combo_df['emp_ttl_comm'] = combo_df['comm_pct'] * combo_df['ttl_sales']
print(combo_df)
# Why are they making so much? the comm_pct needs to be a percent. 
# Fix the employee payout calculation by turning the comm_pct column into an actual percent:



######################