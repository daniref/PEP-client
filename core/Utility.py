import requests
import json
from urllib.parse import urljoin
import csv
import os
import xml.etree.ElementTree as ET
import math 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.stats import norm
import seaborn as sns

# Configuration for the server addresses
ServerAddress = ""
PrivServerAddress = "" # Insert here your Private server address
PubServerAddress = "" # Insert here your Public server address

# Paths for the server endpoints
httpGetDevAv = "/devices/available"
httpGetPower = "/power"
httpPostConfigFile = '/config'
httpGetUserData = '/user'
httpExpCampaignDownloading = '/results'
httpInfoExps = '/numExps'
httpLogIn = '/login'

# Configuration names for different files
pufsConfigName='pufsConfig'
bitstreamName='bitstream'
expsConfigName='expsConfig'

# Directory paths for test files and results
baseTestsDir='tests/test'

certFile='core/cert.pem'


# ANSI color codes
CYAN = "\033[96m"
WHITE = "\033[97m"
RESET = "\033[0m"
GREEN = "\033[92m"
RED = "\033[91m"

# Function to display the SPECTRE logo with colors
def DisplayLogo():
    print(f"""
  .-')     _ (`-.    ('-.             .-') _   _  .-')     ('-.   
 ( OO ).  ( (OO  ) _(  OO)           (  OO) ) ( \( -O )  _(  OO)  
(_)---\_)_.`     \(,------.   .-----./     '._ ,------. (,------. 
/    _ |(__...--'' |  .---'  '  .--./|'--...__)|   /`. ' |  .---' 
\  :` `. |  /  | | |  |      |  |('-.'--.  .--'|  /  | | |  |     
 '..`''.)|  |_.' |(|  '--.  /_) |OO  )  |  |   |  |_.' |(|  '--.  
.-._)   \|  .___.' |  .--'  ||  |`-'|   |  |   |  .  '.' |  .--'  
\       /|  |      |  `---.(_'  '--'\   |  |   |  |\  \  |  `---. 
 `-----' `--'      `------'   `-----'   `--'   `--' '--' `------'' 
 """)
    print(f"""{GREEN}System for PUFs Evaluation{RESET}, Characterization, Testing, {RED} and Reliability Estimation.{RESET}
    """)

def SetServerAddress(serverAddress):
    global ServerAddress
    if serverAddress == 'pub':
        ServerAddress =  PubServerAddress 
        return True
    elif serverAddress == 'priv':
        ServerAddress =  PrivServerAddress
        return True
    else:
        print("[CLIENT-APP] - Server choice not recognized")
        return False

def GetServerAddress():
    return ServerAddress

