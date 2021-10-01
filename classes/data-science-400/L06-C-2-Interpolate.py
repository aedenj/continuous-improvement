# UW Data Science
# Please run code snippets one at a time to understand what is happening.
# Snippet blocks are sectioned off with a line of ####################

import pandas as pd

# you can have pandas convert the date directly during the import
# Be patient, this step will take some processing time (30+seconds depending on internet connection)
pm2_file = pd.read_csv("https://library.startlearninglabs.uw.edu/DATASCI400/Datasets/BeijingPM2_IOT.csv",
                       parse_dates=['TimeStamp'],
                       infer_datetime_format=True) 
#########################

# chained subset for temp, precip, humidity
pm2_df = pm2_file[(pm2_file['Attribute'] == 'precipitation') | 
        (pm2_file['Attribute'] == 'TEMP') | 
        (pm2_file['Attribute'] == 'HUMI')].copy()

# show a sample of the data
print(pm2_df.head())

##########################

# set time as an index so we can easily group data by date
pm2_df = pm2_df.set_index(['TimeStamp'])

# ensure that integer column is correct data type
pm2_df['Value'] = pm2_df['Value'].astype(float)

# drop rows with missing values
pm2_df = pm2_df.dropna(axis=0, how='any')
##########################

# group data by attribute
grouped = pm2_df.groupby('Attribute')

# grouped is now a groupby object
print(grouped)

# all the same 'Attribute' strings are grouped together
for x,y in grouped:
    print(x)
    print(y)
# how many HUMI readings are there?
# how many TEMP readings are there?
# how many precipitation readings are there?
##########################

# group multiple columns/objects such as attribute and year
year_attr_group = pm2_df.groupby([pm2_df.index.year, pm2_df.Attribute])

for x,y in year_attr_group:
    print(x)
    print(y)
# How many HUMI readings are in the year 2011?
##########################
    
# resample to 1 minute intervals
downsample = pm2_df.resample('1min')
print(downsample.head(100))    

# This step also takes some time(up to 3minutes depending on RAM). Watch your memory usage.
upsampled = pm2_df.groupby('Attribute').resample('1S').mean()

for group, x in upsampled.iteritems():
    print(group)
    print(x)
# How many readings are there now?
#########################

# upsampled data, fill in NaN by interpolation
upsampled_interpolated = upsampled.interpolate(method='linear')

#########################
#show sample of the data, what has changed?
print(upsampled_interpolated.head())

# compare plots of HUMI data
upsampled_humi = upsampled_interpolated.loc['HUMI']

# The following line code will take a long time to run. Please be patient.
upsampled_humi.plot() ##Potential memory error if you have less than 8Gig RAM

#########################
humidity = pm2_df[(pm2_df['Attribute'] == 'HUMI')]
humidity.plot()

#########################
# Interpolate the Temperature Readings
# What happens if you change the method to polynomial?

# Plot the temperature data

# Are the distributions different?

########################