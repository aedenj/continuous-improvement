#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 17:07:59 2019

@author: aedenjameson

Description: Nothing terribly exciting. We're just going to
print out the number of links on a webpage
"""

import requests
from bs4 import BeautifulSoup


url = "http://archive.ics.uci.edu/ml/datasets.php"
response = requests.get(url)
soup = BeautifulSoup(response.content, "lxml")

all_a = soup.find_all("a")

print(f'The webpage {url} has {len(all_a)} links')