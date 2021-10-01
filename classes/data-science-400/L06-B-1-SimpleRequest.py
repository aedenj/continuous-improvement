"""
# UW Data Science
# To run the code, hilight lines by selecting the line number
# and then press CTRL+ENTER or choose Run Selection from the toolbar.
"""
# Load the package
import requests 

# Grab data from an external URL
response = requests.get("https://en.wikipedia.org/robots.txt") 

# Assign the text of the file to a variable
txt = response.text 

# Print the text portion of the file
print(txt)

#########################