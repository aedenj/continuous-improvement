"""
# UW Data Science
# Please run code snippets one at a time to understand what is happening.
# Snippet blocks are sectioned off with a line of ####################
"""

import json 

# create JSON array
json_array = '{"drinks": ["coffee", "tea", "water"]}'
# How many elements are in this array? Check the Variable explorer.

#######################
# convert JSON array to python list with json.loads()
data = json.loads(json_array ) 

# loop through list items
for element in data['drinks']:    
    print(element)

#######################