"""
# UW Data Science
# Please run code snippets one at a time to understand what is happening.
# Snippet blocks are sectioned off with a line of ####################
"""

# lets import the libraries we are going to use
import requests
from bs4 import BeautifulSoup 

# then we'll set up the URL we want to pull from as a string
url = "https://wiki.python.org/moin/IntroductoryBooks" 
 
# now we can use urllib to pull down the html and store it as a request object
response = requests.get(url)

# This gives us back a requests object that allows us to pull out different parts 
# of the web page for parsing using beautifulsoup.

# here can view the page headers
print(response.headers)
###########################

# but what we want is to grab the page content
content = response.content

print(content)
###########################

# now that we've pulled down the page content, let's use beautifulsoup to 
# convert it to something that we can read, here we add the lxml tag 
# to notify beautifulsoup about what type of HTML format we are working with
soup = BeautifulSoup(content, "lxml")

# we can use BeautifulSoup's prettify function to print the html from our soup object in a more readable format 
# so that we can figure out what to grab out
print(soup.prettify())
###########################

# now we can see the layout of nested tags, which helps us figure out
# how to get the information we need

# for example, if we want to get the information insided title tags we could do: 
print(soup.title)

# We can even convert this to a string for downstream use
print(soup.title.string)
#############################

# if we want to view information contained in certain tags, we can use the find_all function: 
# let's say we want to grab all info inside a tags, which commonly contains links 
all_a = soup.find_all("a")

# this returns in iterable object that we can loop through
for x in all_a:
    print(x)
##############################
    
# We can fine tune this information even further by adding a class argument to the tag
# for example, if want only web links, we can look inside "a" tags for the "http" tags: 
all_a_https = soup.find_all("a", "https")   

for x in all_a_https:
    print(x) 
##############################
    
# we can access items inside the iterable just like with a regular python list
print(all_a_https[0])

# and we can even convert that result diretly to a string
print(all_a_https[0].string)
##############################

#We can also loop through other the metadata (attributes, just like variables) 
# that are nested inside of the dev tag that we pulled out: 

for x in all_a_https:
    print(x.attrs['href'])
    
# Now that we know how to pull out data, we can pull out the elements we need
# and automatically convert them into useful python data stucutres, like a dictionary: 
    
data = {}    
for a in all_a_https: 
    title = a.string.strip() 
    data[title] = a.attrs['href']

print(data)
##############################