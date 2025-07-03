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
# Directory path for certificate file
certFile='core/cert.pem'

# ANSI color codes
CYAN = "\033[96m"
WHITE = "\033[97m"
RESET = "\033[0m"
GREEN = "\033[92m"
RED = "\033[91m"

def DisplayLogo():
    """Displays the SPECTRE logo in the console with color formatting.
    """
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
    """
    Sets the server address based on the user's choice.
    Args:
        serverAddress (str): 'pub' for public server, 'priv' for private server.
    Returns:
        bool: True if the server address was set successfully, False otherwise.
    """
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
    """
    Returns the current server address.
    """
    return ServerAddress

