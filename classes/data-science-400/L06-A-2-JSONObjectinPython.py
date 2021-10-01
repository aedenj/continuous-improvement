# -*- coding: utf-8 -*-
"""
# UW Data Science
# Please run code snippets one at a time to understand what is happening.
# Snippet blocks are sectioned off with a line of ####################
"""
"""
Example script for JSON assessor methods
"""

import json

# create JSON object
json_data = '{"name":"Steven", "city":"Seattle"}'
#notice the new variable in the explorer.
####################

# convert JSON object to python dictionary with json.loads()
python_obj = json.loads(json_data)
# notice the new variable in the explorer
####################

# print dictionary values by keys
print(python_obj["name"])

print(python_obj["city"])
##################