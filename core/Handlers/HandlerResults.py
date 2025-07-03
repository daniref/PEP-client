import os

from core.Utility import *
from core.Metrics import Parser as PS
from core.Metrics import Metrics as MT

# oscillPeriod represents the oscillation period in which the ro-PUF works. 
# It is used to calculate the frequencies of the counter1 and counter2 values in the CRP files.
oscillPeriod = 0.01 

def ExpCampaignDownloading(idTest,username,password):
    """
    Download the experiment campaigns for a given test ID and save them as CSV files.
    Args:
        idTest (int): The ID of the test for which to download the campaigns.
        username (str): The username of the user.
        password (str): The password of the user.
    """
    idCampaignList = _ReadIdCampaigns(idTest)

    for idCampaign in idCampaignList:
        httpUrl = urljoin(GetServerAddress(), httpExpCampaignDownloading)
        data = {'idCampaign': idCampaign,'username': username, 'password': password}
        response = requests.post(httpUrl, data=data, verify=certFile)

        # Check if the response is successful
        if response.status_code == 200:
            # we assume the response is in JSON format and contains the fields 'csv', 'field', and 'line'
            try:
                responseData = response.json()
                csvData = responseData['csv']
                fieldDelimiter = responseData['field']
                lineDelimiter = responseData['line']
            except (KeyError, ValueError) as e:
                print(f"Errore nell'interpretazione della risposta per idCampaign {idCampaign}: {e}")
                continue

            resultsDirectory=baseTestsDir+str(idTest)+'/results'
            if not os.path.exists(resultsDirectory):
                os.makedirs(resultsDirectory)

            filePath = f"{resultsDirectory}/expCampaign_{idCampaign}.csv"

            # Save the CSV data to a file
            try:
                with open(filePath, mode='w', newline='') as csvFile:
                    writer = csv.writer(csvFile, delimiter=fieldDelimiter, lineterminator=lineDelimiter)
                    for row in csvData.split(lineDelimiter):
                        writer.writerow(row.split(fieldDelimiter))
                print(f"File salvato con successo in {filePath} per idCampaign {idCampaign}")
            except IOError as e:
                print(f"Errore durante il salvataggio del file per idCampaign {idCampaign}: {e}")
        else:
            # Log the error with the status code
            print(f"[CLIENT-APP] - Request Exp Campaign failed for idCampaign {idCampaign} with code {response.status_code}")

def _ReadIdCampaigns(idTest):
    """
    Reads the idCampaigns from the campaigns.csv file for a given test ID.
    Args:
        idTest (int): The ID of the test for which to read the campaigns.
    Returns:
        list: A list of idCampaigns.
    """
    testDir = baseTestsDir+str(idTest)+'/'
    filePath = os.path.join(testDir, 'campaigns.csv')
    
    # Check if the file exists
    if not os.path.isfile(filePath):
        print(f"Il file {filePath} non esiste.")
        return []
    
    # Read the CSV file using pandas
    try:
        df = pd.read_csv(filePath)
        # Return a list of idCampaigns
        return df['idCampaign'].tolist()
    except KeyError:
        print(f"The column 'idCampaign' does not exist in the file.")
        return []

def CheckCRPsExist(idTest):
    """
    Checks if the CRP files for the given test ID exist in the results directory.
    Args:
        idTest (int): The ID of the test for which to check the CRP files.
    Returns:
        bool: True if all required CRP files exist, False otherwise.
    """
    idCampaignsList = _ReadIdCampaigns(idTest)
    resultsDir = baseTestsDir+str(idTest)+'/results'
    
    # Check if the results directory exists
    if not os.path.exists(resultsDir):
        print(f"Il path {resultsDir} non esiste.")
        return False
    
    # Assuming idCampaignsList contains integers, convert them to strings for file naming
    filesToCheckList = {f"expCampaign_{id}.csv" for id in idCampaignsList}
    
    # Use set to avoid duplicates and for efficient membership testing
    filePresent = set(os.listdir(resultsDir))
    
    # Check if all files in filesToCheckList are present in filePresent
    return filesToCheckList.issubset(filePresent)

def _GetTestInfo(idTest):
    """
    Reads the test information from the campaigns.csv and expsConfig.xml files for a given test ID.
    Args:
        idTest (int): The ID of the test for which to read the information.
    Returns:
        tuple: A tuple containing:
            - idPufLists: List of lists of PUF IDs for each experiment.
            - numChallengeList: List of number of challenges used in each experiment.
            - respWidthList: List of response widths for each experiment.
            - numRepsList: List of number of repetitions for each experiment.
            - devicesList: List of device IDs.
            - campaignList: List of campaign IDs.
            - hasCountersList: List of boolean values indicating if counters are used in each experiment.
    """
    testDir = baseTestsDir + str(idTest)

    campaignList = [] # List of campaign ids
    devicesList = [] # List of devices ids
    numChallengeList = [] # List of number of challenges used in each experiment (max= number of different experiments, es 2 rand, 3 list, 4 range)
    respWidthList = [] # List of response width for each experiment (max= number of different experiments)
    numRepsList = [] # List of number of repetitions for each experiment (max= number of different experiments)
    idPufLists = [] # List of list of puf ids for each experiment (max= number of different experiments)
    hasCountersList = []  # List of boolean values for each experiment (max= number of different experiments)

    # Read the campaigns CSV file
    df = pd.read_csv(testDir + '/campaigns.csv')

    # Retrieves the unique device IDs and campaign IDs from the DataFrame
    devicesList = df['idDev'].tolist()
    campaignList = df['idCampaign'].tolist()

    # Read the experiment configurations from the expsConfig.xml file
    tree = ET.parse(testDir + '/expsConfig.xml')
    root = tree.getroot()

    # Reads the PUF configurations from the pufsConfig.xml file
    pufTree = ET.parse(testDir + '/pufsConfig.xml')
    pufRoot = pufTree.getroot()
    pufInstances = pufRoot.findall("PUFInstance")

    # Cycle through each experiment element in the XML
    for expElement in root:
        expType = expElement.tag  # Get the type of experiment (e.g., RangeExp, ListExp, RandomExp)
        numExps = expElement.find("num_exps")
        numRepsList.append(int(numExps.text))  # Convert the number of experiments to an integer and add it to the list
        if expType == "RangeExp": # Handles the RangeExp experiment type
            challengeWidth = int(expElement.find("challenge_bits_width").text)
            step = int(expElement.find("step").text)
            numChal = math.floor((2 ** challengeWidth) / step)
            numChallengeList.append(numChal)

        elif expType == "ListExp": # Handles the ListExp experiment type
            challengesList = expElement.find("challenges_list")
            numChal = len(challengesList.findall("challenge"))
            numChallengeList.append(numChal)

        elif expType == "RandomExp": # Handles the RandomExp experiment type
            numChalElement = expElement.find("num_challenges")
            numChallengeList.append(int(numChalElement.text))

        # List of PUF IDs
        idsList = expElement.find("puf_ids")
        ids = [int(idElement.text) for idElement in idsList.findall("id")]  # Convert each ID to an integer
        idPufLists.append(ids)  # Add the list of PUF IDs to the main list

        # Add the response width to the list
        respWidth = pufInstances[ids[0]].find("respSize")
        respWidthList.append(int(respWidth.text))  # Convert the response width to an integer and add it to the list

        # Check if the PUF has counters
        countSize = pufInstances[ids[0]].find("countSize")
        if countSize is None or int(countSize.text) == 0:
            hasCounters = False  # if there is no countSize or it is 0, then there are no counters
        else:
            hasCounters = True
        hasCountersList.append(hasCounters)

    return idPufLists, numChallengeList, respWidthList, numRepsList, devicesList, campaignList, hasCountersList

def CalculateMetrics(idTest):
    """
    Calculates various metrics for the given test ID by reading the experiment data and computing metrics like uniformity, reliability, bit aliasing, and uniqueness.
    Args:
        idTest (int): The ID of the test for which to calculate metrics.
    """
    idPufLists,numChallengeList,ResponseWidthList,NumRepsList,devicesList,campaignList,hasCountersList=_GetTestInfo(idTest)
    print(idPufLists,numChallengeList,ResponseWidthList,NumRepsList,devicesList,campaignList,hasCountersList)
    resultsCSVsBase = baseTestsDir+str(idTest)+'/results/expCampaign_'

    expChsLists = []
    expCRPsLists=[]
    expTemperatureList = []
    # map of device IDs to indices and vice versa
    idDevToIndexDev={}
    devicesValues=set()
    for idDev in devicesList:
        devicesValues.add(idDev)
    idDevToIndexDev = {val: idx for idx, val in enumerate(sorted(devicesValues))}
    indexDevToIdDev = {idx: val for idx, val in enumerate(sorted(devicesValues))}
    # each cycle is a test of type list, range or random on a type of PUF
    # each of these tests may have been executed on multiple PUF instances on the same device
    # it has certainly been executed on different devices
    for indexExp, idPufList in enumerate(idPufLists): #idPufLists is a list of lists of PUF IDs for each experiment

        expCRPsList=[]
        expChsList = []
        
        # the challenge list is the same for all PUFs in the same experiment
        csvPath = f"{resultsCSVsBase}{campaignList[0]}.csv"
        df = pd.read_csv(csvPath)
        challengeValue = set()
        challengeValue = sorted(df[df['idpuf'].isin(idPufList)]['challenge'].unique())

        expChsList.append(challengeValue)
        challengeMap = {val: idx for idx, val in enumerate(sorted(expChsList[indexExp]))}


        print(f"devicesList: {devicesList}")

        # list of unique devicesList
        unidevicesList = list(set(devicesList))
        # this cycle checks the number of PUF instances on which the test has been executed
        for idPuf in idPufList: 
            
            crpsNpArray = np.full((numChallengeList[indexExp],ResponseWidthList[indexExp],NumRepsList[indexExp],len(unidevicesList)),np.nan) # le ultime dim sono FredDIff e Temperature
            TemperatueArray = np.full((numChallengeList[indexExp],NumRepsList[indexExp],len(unidevicesList)),-300) # le ultime dim sono FredDIff e Temperature

            expCRPsList.append(crpsNpArray)
            print("size of crpsNpArray", crpsNpArray.shape)
            print("devicesList: ", devicesList)
            # cycle on each device (identified by idDev) for the specific campaign
            for index, idDev in enumerate(devicesList):
                # Reads the CSV file for the specific campaign of the device
                csvPath = f"{resultsCSVsBase}{campaignList[index]}.csv"
                df = pd.read_csv(csvPath,dtype={"response": str})
                
                # Filter the rows for the specific PUF
                filteredPuf = df[df['idpuf'] == idPuf]
                
                # obtain the index of the device in the crpsNpArray
                idDevIdx = idDevToIndexDev[idDev]
                
                # Populate crpsNpArray with data from the CSV
                for _, row in filteredPuf.iterrows():
                    challengeVal = row['challenge']
                    challengeIdx = challengeMap[challengeVal]  # Obtain the index of the challenge
                
                    responseHex = row['response']
                    temperature = row['temperature']
                    response = int(responseHex, 16)
                    bitstringResp = f'{response:0{ResponseWidthList[indexExp]}b}'  # Convert to binary string with leading zeros
                    repIdx = int(row['numrep'])
                    # Insert the temperature and response bits into the respective arrays
                    TemperatueArray[challengeIdx, repIdx, idDevIdx] = temperature
                    for bitIdx, bit in enumerate(bitstringResp):
                        crpsNpArray[challengeIdx, bitIdx, repIdx, idDevIdx] = int(bit)

        expCRPsLists.append(expCRPsList)
        expTemperatureList.append(TemperatueArray)

        for i in range(len(expCRPsList)):
            MT.ComputeUniformity(idTest,expCRPsList[i],i,indexDevToIdDev)
            MT.ComputeReliability(idTest,expCRPsList[i],i,indexDevToIdDev)
            MT.ComputeBitAliasing(idTest,expCRPsList[i],i,indexDevToIdDev)
            MT.ComputeUniqueness(idTest,expCRPsList[i],i,indexDevToIdDev)
        
        if hasCountersList[indexExp]:
            for index, idDev in enumerate(devicesList):
                # Read the CSV file for the specific campaign of the device
                csvPath = f"{resultsCSVsBase}{campaignList[index]}.csv"
                df = _ExtractRelevantFields(csvPath)
                MT.AnalyzeFrequencies(idTest,idDev,idPufList,df,oscillPeriod)

def _ExtractRelevantFields(csvFilePath):
    """
    Extracts relevant fields from a CSV file and returns a DataFrame with the required columns.
    Args:
        csvFilePath (str): The path to the CSV file.
    Returns:
        pd.DataFrame: A DataFrame containing the relevant fields.
    """
    df = pd.read_csv(csvFilePath)

    # Checks if the required columns are present in the DataFrame
    required_columns = ['idpuf', 'numrep', 'challenge', 'response', 'temperature', 'counter1', 'counter2']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Le seguenti colonne sono mancanti nel CSV: {', '.join(missing_columns)}")

    # Filter only the required columns
    filtered_df = df[required_columns]

    # Delete rows where counter1 or counter2 are NaN
    filtered_df = filtered_df.dropna(subset=['counter1', 'counter2'])

    return filtered_df