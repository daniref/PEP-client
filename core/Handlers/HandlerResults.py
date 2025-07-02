import os

from core.Utility import *
from core.Metrics import Parser as PS
from core.Metrics import Metrics as MT

oscillPeriod = 0.01 # It represents the oscillation period in which the ro-PUF works

def ExpCampaignDownloading(idTest,username,password):
    
    idCampaignList = _ReadIdCampaigns(idTest)

    for idCampaign in idCampaignList:
        httpUrl = urljoin(GetServerAddress(), httpExpCampaignDownloading)
        data = {'idCampaign': idCampaign,'username': username, 'password': password}
        response = requests.post(httpUrl, data=data, verify=certFile)

        # Verifica se la richiesta è andata a buon fine (codice di stato 200)
        if response.status_code == 200:
            # Presupponiamo che la risposta sia in formato JSON e contenga i campi 'csv', 'field' e 'line'
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

            # Salva il CSV nel file
            try:
                with open(filePath, mode='w', newline='') as csvFile:
                    writer = csv.writer(csvFile, delimiter=fieldDelimiter, lineterminator=lineDelimiter)
                    for row in csvData.split(lineDelimiter):
                        writer.writerow(row.split(fieldDelimiter))
                print(f"File salvato con successo in {filePath} per idCampaign {idCampaign}")
            except IOError as e:
                print(f"Errore durante il salvataggio del file per idCampaign {idCampaign}: {e}")
        else:
            # Se la richiesta non è andata a buon fine, gestisci l'errore
            print(f"[CLIENT-APP] - Request Exp Campaign failed for idCampaign {idCampaign} with code {response.status_code}")

def GetInfoExps(idCampaign,username,password):
    httpUrl = urljoin(GetServerAddress(), httpInfoExps)
    data={'idcampaign': idCampaign,'username': username, 'password': password}
    response = requests.post(httpUrl,data=data, verify=certFile)

    # Verifica se la richiesta è andata a buon fine (codice di stato 200)
    if response.status_code == 200:
        print('[CLIENT-APP] - GetInfoExps() - Obtained Info about Exp Campaign ',idCampaign)
        return response.json()['num_exps']
    else:
        # Se la richiesta non è andata a buon fine, gestisci l'errore
        print('[CLIENT-APP] - GetInfoExps() - Request for info about Exp failed ', response.status_code)

def _ReadIdCampaigns(idTest):
    testDir = baseTestsDir+str(idTest)+'/'
    # Path del file csv
    filePath = os.path.join(testDir, 'campaigns.csv')
    
    # Verifica se il file esiste
    if not os.path.isfile(filePath):
        print(f"Il file {filePath} non esiste.")
        return []
    
    # Leggi il file csv utilizzando pandas
    try:
        df = pd.read_csv(filePath)
        # Restituisci la lista degli idCampaign
        return df['idCampaign'].tolist()
    except KeyError:
        print(f"La colonna 'idCampaign' non esiste nel file.")
        return []

def CheckCRPsExist(idTest):
    """
    Checks if CRP (Campaign Result Package) files exist for a given test campaign.
    @param idTest: The ID of the test to check for CRP file existence.
    @type idTest: int
    @param idCampaignsList: List of campaign IDs to check for CRP file existence.
    @type idCampaignsList: list[int]
    @return: True if all required CRP files exist, False otherwise.
    @rtype: bool
    """
    idCampaignsList = _ReadIdCampaigns(idTest)
    resultsDir = baseTestsDir+str(idTest)+'/results'
    
    # Verifica se il path esiste
    if not os.path.exists(resultsDir):
        print(f"Il path {resultsDir} non esiste.")
        return False
    
    # Set con gli ID richiesti trasformati in stringhe (per confronti diretti)
    filesToCheckList = {f"expCampaign_{id}.csv" for id in idCampaignsList}
    
    # Set con i nomi dei file presenti nella directory
    filePresent = set(os.listdir(resultsDir))
    
    # Verifica se tutti i file richiesti sono presenti
    return filesToCheckList.issubset(filePresent)

def GetTestInfo(idTest):
    testDir = baseTestsDir + str(idTest)

    campaignList = [] # List of campaign ids
    devicesList = [] # List of devices ids
    numChallengeList = [] # List of number of challenges used in each experiment (max= number of different experiments, es 2 rand, 3 list, 4 range)
    respWidthList = [] # List of response width for each experiment (max= number of different experiments)
    numRepsList = [] # List of number of repetitions for each experiment (max= number of different experiments)
    idPufLists = [] # List of list of puf ids for each experiment (max= number of different experiments)
    hasCountersList = []  # List of boolean values for each experiment (max= number of different experiments)

    # Leggi il file campaigns CSV
    df = pd.read_csv(testDir + '/campaigns.csv')

    # Legge le occorrenze di idDev
    devicesList = df['idDev'].tolist()
    campaignList = df['idCampaign'].tolist()

    # Leggi le configurazioni degli esperimenti nel file expsConfig.xml
    tree = ET.parse(testDir + '/expsConfig.xml')
    root = tree.getroot()

    # Leggi le informazioni sui PUF dal file pufsConfig.xml
    pufTree = ET.parse(testDir + '/pufsConfig.xml')
    pufRoot = pufTree.getroot()
    pufInstances = pufRoot.findall("PUFInstance")

    # Itera su ogni elemento figlio di root e gestisce in base al tipo di esperimento
    for expElement in root:
        expType = expElement.tag  # Ottieni il tipo di esperimento (es. "RangeExp", "ListExp", "RandomExp")

        numExps = expElement.find("num_exps")
        numRepsList.append(int(numExps.text))  # Converte in intero e aggiunge alla lista

        if expType == "RangeExp":
            # Gestisce l'esperimento di tipo RangeExp
            challengeWidth = int(expElement.find("challenge_bits_width").text)
            step = int(expElement.find("step").text)
            numChal = math.floor((2 ** challengeWidth) / step)
            numChallengeList.append(numChal)

        elif expType == "ListExp":
            # Gestisce l'esperimento di tipo ListExp
            challengesList = expElement.find("challenges_list")
            numChal = len(challengesList.findall("challenge"))
            numChallengeList.append(numChal)

        elif expType == "RandomExp":
            # Gestisce l'esperimento di tipo RandomExp
            numChalElement = expElement.find("num_challenges")
            numChallengeList.append(int(numChalElement.text))

        # Lista di ID dei PUF
        idsList = expElement.find("puf_ids")
        ids = [int(idElement.text) for idElement in idsList.findall("id")]  # Converte in interi
        idPufLists.append(ids)  # Aggiunge la lista di ID a idPufLists

        # Verifica e accoda i valori di respSize e countSize
        respWidth = pufInstances[ids[0]].find("respSize")
        respWidthList.append(int(respWidth.text))  # Converte in intero e aggiunge alla lista

        # Controllo del valore del contatore (deve essere maggiore di 0)
        countSize = pufInstances[ids[0]].find("countSize")
        if countSize is None or int(countSize.text) == 0:
            hasCounters = False  # Se un contatore è 0, aggiorna il flag
        else:
            hasCounters = True
        hasCountersList.append(hasCounters)

    return idPufLists, numChallengeList, respWidthList, numRepsList, devicesList, campaignList, hasCountersList

def CalculateMetrics(idTest):
    idPufLists,numChallengeList,ResponseWidthList,NumRepsList,devicesList,campaignList,hasCountersList=GetTestInfo(idTest)
    print(idPufLists,numChallengeList,ResponseWidthList,NumRepsList,devicesList,campaignList,hasCountersList)
    resultsCSVsBase = baseTestsDir+str(idTest)+'/results/expCampaign_'

    expChsLists = []
    expCRPsLists=[]
    expTemperatureList = []
    # mapping di idDev a indexDev e viceversa
    idDevToIndexDev={}
    devicesValues=set()
    for idDev in devicesList:
        devicesValues.add(idDev)
    idDevToIndexDev = {val: idx for idx, val in enumerate(sorted(devicesValues))}
    indexDevToIdDev = {idx: val for idx, val in enumerate(sorted(devicesValues))}
    # ogni ciclo è un test di tipo list, range o random su una tipologia di puf
    # ognuno di questi test può essere stato eseguito su più istanze di puf sullo stesso dispositivo
    #sicuramente è stato eseguito su dispositivi diversi
    for indexExp, idPufList in enumerate(idPufLists): #idPufLists è una lista di liste di idPuf
        
        expCRPsList=[]
        expChsList = []
        
        # la lista challenge è uguale per tutte le istanze di puf del medesimo test
        csvPath = f"{resultsCSVsBase}{campaignList[0]}.csv"
        df = pd.read_csv(csvPath)
        challengeValue = set()
        challengeValue = sorted(df[df['idpuf'].isin(idPufList)]['challenge'].unique())

        # challengeValue = sorted(df[df['idpuf'] == idPufList[0]]['challenge'].unique())
        expChsList.append(challengeValue)
        challengeMap = {val: idx for idx, val in enumerate(sorted(expChsList[indexExp]))}


        print(f"devicesList: {devicesList}")

        #lista di devicesList unici
        unidevicesList = list(set(devicesList))
        #questo ciclo controlla il numero di istanze di puf su cui è stato eseguito il test
        for idPuf in idPufList: 
            
            crpsNpArray = np.full((numChallengeList[indexExp],ResponseWidthList[indexExp],NumRepsList[indexExp],len(unidevicesList)),np.nan) # le ultime dim sono FredDIff e Temperature
            TemperatueArray = np.full((numChallengeList[indexExp],NumRepsList[indexExp],len(unidevicesList)),-300) # le ultime dim sono FredDIff e Temperature

            expCRPsList.append(crpsNpArray)
            print("size of crpsNpArray", crpsNpArray.shape)
            print("devicesList: ", devicesList)
            # Itera su ogni dispositivo (identificato da idDev) per la campagna specifica
            for index, idDev in enumerate(devicesList):
                # Legge il CSV per la campagna specifica del dispositivo
                csvPath = f"{resultsCSVsBase}{campaignList[index]}.csv"
                df = pd.read_csv(csvPath,dtype={"response": str})
                
                # Filtra le righe per la PUF specifica
                filteredPuf = df[df['idpuf'] == idPuf]
                
                # Ottieni l'indice del dispositivo dalla mappatura
                idDevIdx = idDevToIndexDev[idDev]
                
                # Popola crpsNpArray con i dati dal CSV
                for _, row in filteredPuf.iterrows():
                    challengeVal = row['challenge']
                    challengeIdx = challengeMap[challengeVal]  # Ottieni l'indice mappato
                
                    responseHex = row['response']
                    temperature = row['temperature']
                    response = int(responseHex, 16)
                    bitstringResp = f'{response:0{ResponseWidthList[indexExp]}b}'  # Conversione a binario con padding
                    repIdx = int(row['numrep'])
                    # Inserisci ogni bit della risposta nella posizione corretta
                    #print(f"size crpsNpArray {crpsNpArray.shape}")
                    TemperatueArray[challengeIdx, repIdx, idDevIdx] = temperature
                    for bitIdx, bit in enumerate(bitstringResp):
                        crpsNpArray[challengeIdx, bitIdx, repIdx, idDevIdx] = int(bit)

        expCRPsLists.append(expCRPsList)
        expTemperatureList.append(TemperatueArray)

        for i in range(len(expCRPsList)):
            MT.ComputeUniformity(idTest,expCRPsList[i],i,indexDevToIdDev)
            # MT.ComputeReliability(idTest,expCRPsList[i],i,indexDevToIdDev)
            # MT.ComputeBitAliasing(idTest,expCRPsList[i],i,indexDevToIdDev)
            # MT.ComputeUniqueness(idTest,expCRPsList[i],i,indexDevToIdDev)
            # MT.ComputeBitReliability(idTest,expCRPsList[i],i,indexDevToIdDev)
            # MT.ComputeMinHentropyDensity(idTest,expCRPsList[i],i,indexDevToIdDev)
            # MT.ComputeTemperatureReliability(idTest,expCRPsList[i],i,indexDevToIdDev,TemperatueArray)
        
        if hasCountersList[indexExp]:
            for index, idDev in enumerate(devicesList):
                # Legge il CSV per la campagna specifica del dispositivo
                csvPath = f"{resultsCSVsBase}{campaignList[index]}.csv"
                df = ExtractRelevantFields(csvPath)
                MT.AnalyzeFrequencies(idTest,idDev,idPufList,df,oscillPeriod)


def ExtractRelevantFields(csv_file_path):
    """
    Legge un CSV, filtra i campi richiesti ed elimina le righe con counter1 o counter2 vuoti.

    Parameters:
    - csv_file_path: str, il percorso al file CSV.

    Returns:
    - filtered_df: pd.DataFrame, un DataFrame contenente solo i campi richiesti e senza righe con counter1 o counter2 vuoti.
    """
    # Leggi il CSV
    df = pd.read_csv(csv_file_path)

    # Controlla se le colonne richieste esistono nel CSV
    required_columns = ['idpuf', 'numrep', 'challenge', 'response', 'temperature', 'counter1', 'counter2']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Le seguenti colonne sono mancanti nel CSV: {', '.join(missing_columns)}")

    # Filtra solo le colonne richieste
    filtered_df = df[required_columns]

    # Elimina le righe in cui counter1 o counter2 sono vuoti
    filtered_df = filtered_df.dropna(subset=['counter1', 'counter2'])

    return filtered_df