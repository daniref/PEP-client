import os

from core.Utility import *
import core.Handlers.HandlerAccess as HA

def _GetDevicesAvailable(username,password):
    data = {
        'username':username,
        'password':password
    }
    httpUrl = urljoin(GetServerAddress(), httpGetDevAv)
    # Effettua una richiesta GET al server
    response = requests.post(httpUrl, data=data, verify=certFile)

    # Verifica se la richiesta è andata a buon fine (codice di stato 200)
    if response.status_code == 200:
        # La risposta contiene i dispositivi disponibili, quindi possiamo accedere ai dati
        devices = response.json()  # Converti la risposta in formato JSON
        return devices
    else:
        # Se la richiesta non è andata a buon fine, gestisci l'errore
        print('[CLIENT-APP] - Http Get: Devs Available Error:', response.status_code)

def _LockDeviceByIdDev(idUser,idDevice,username,password):

    httpUrl = urljoin(GetServerAddress(), httpGetDevAv)
    params={'idUser': idUser, 'idDevice': idDevice, 'state': 'unavailable','username':username,'password':password}
    response = requests.post(httpUrl,data=params, verify=certFile)

    # Verifica se la richiesta è andata a buon fine (codice di stato 200)
    if response.status_code == 200:
        # La risposta contiene i dispositivi disponibili, quindi possiamo accedere ai dati
        resultOperation = response.json()  # Converti la risposta in formato JSON
        return resultOperation
    else:
        # Se la richiesta non è andata a buon fine, gestisci l'errore
        print('[CLIENT-APP] - Http Get: Lock Device Error:', response.status_code)

def _SendsPufsConfig(idUser,testDir,username,password):

    #send pufs conf xml
    httpUrl = urljoin(GetServerAddress(), httpPostConfigFile)
    file = {'file': open(testDir+pufsConfigName+'.xml', 'rb')}
    payload = {'idUsr':idUser,'type':'pufsConf','username':username,'password':password}   
    response = requests.post(httpUrl,files=file,data=payload,verify=certFile) 

    idPufsConfig = None
    if response.status_code == 200:
        if(response.json().get('result') == True):
            idPufsConfig = response.json().get('idpufsconfig')
            print("[CLIENT-APP] - LaunchTests() - xml puf config correctly sent with id ", idPufsConfig)
            httpUrl = urljoin(GetServerAddress(), httpPostConfigFile)
            file = {'file': open(testDir+bitstreamName+'.bin', 'rb')}
            payload = {'idUsr':idUser,'type':'bitstream', 'idpufsconfig':idPufsConfig,'username':username,'password':password}   
            response = requests.post(httpUrl,files=file,data=payload,verify=certFile)
            if response.status_code == 200:
                if response.json().get('result')==True:
                    print("[CLIENT-APP] - LaunchTests() - bitstream correctly sent associated with id puf config", idPufsConfig)
                    return True,idPufsConfig
            else:
                print("[CLIENT-APP] - LaunchTests() - bitstream NOT correctly sent, error", response.status_code)
    return False, None
 
def _SendsExpsConfig(idUser,idDevice,idPufsConfig,testDir,username,password):

    #send exp json
    httpUrl = urljoin(GetServerAddress(), httpPostConfigFile)
    file = {'file': open(testDir+expsConfigName+'.xml', 'rb')}
    payload = {'idUsr':idUser,'idDev':idDevice,'type':'expsConf', 'idpufsconfig':idPufsConfig,'username':username,'password':password}   
    response = requests.post(httpUrl,files=file,data=payload,verify=certFile) 
    idCampaign = None
    if response.status_code == 200:
        if(response.json().get('result') == True):
            idCampaign = response.json().get('idcampaign')
            return True,idCampaign
    return False, None

def _LockAndProgramDevice(idUser,idDevice,idPufsConfig,testDir,username,password):
    resOp = _LockDeviceByIdDev(idUser,idDevice,username,password) #blocco il primo utile
    if resOp:
        print('[CLIENT-APP] - LockAndProgramDevice() - Device with id ',idDevice,' locked successfully!')
        # Inizia a comandare il device

        # program FPGA
        result,idcamp = _SendsExpsConfig(idUser,idDevice,idPufsConfig,testDir,username,password)
        if(result==True):
            print('[CLIENT-APP] - LockAndProgramDevice() - Device with id ',idDevice,' correctly programmed with id campaign ',idcamp)
            _SaveCampaignData(idcamp,idDevice,testDir)
        else:
            print('[CLIENT-APP] - LockAndProgramDevice() - Device with id ',idDevice,' not correctly programmed!')
    else:
        print('[CLIENT-APP] - LockAndProgramDevice() - Device with id ',idDevice,' not correctly locked!')

def _SaveCampaignData(idCampaign, idDevice, testDirectory):
    # Assicurati che la directory esista
    if not os.path.exists(testDirectory):
        os.makedirs(testDirectory)
    
    # Path del file csv
    filePath = os.path.join(testDirectory, 'campaigns.csv')
    
    # Verifica se il file esiste, in modo da scrivere l'header solo una volta
    fileExists = os.path.isfile(filePath)
    
    # Apri il file in modalità append per aggiungere nuovi dati
    with open(filePath, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Se il file non esiste, scrivi l'header
        if not fileExists:
            writer.writerow(['idCampaign', 'idDev'])
        
        # Scrivi i dati della campagna
        writer.writerow([idCampaign, idDevice])

def LaunchTests(numDevices,idTest,idUser,username,password):
    testDir = baseTestsDir+str(idTest)+'/'
    print('[CLIENT-APP] - LaunchTests() - Request to program ', numDevices, 'devices!')
    # user not registered
    if idUser:
        print('[CLIENT-APP] - LaunchTests() - User registered with id: ', idUser)

        devicesList = _GetDevicesAvailable(username,password) #chiedo se ci sono device disponibili
        if devicesList is not None:
            print('[CLIENT-APP] - LaunchTests() - Number of available devices: ',len(devicesList))
        else:
            print('[CLIENT-APP] -There are no available devices')
            return
        if len(devicesList) > 0:
            resOp,idPufsConfig = _SendsPufsConfig(idUser,testDir,username,password)
            if resOp == False:
                print('[CLIENT-APP] - LaunchTests() - Error in sending PUF configuration!')
                return
            if numDevices <= len(devicesList):
                for indexDevice in range(numDevices):
                    idDevice = devicesList[indexDevice]['id']
                    _LockAndProgramDevice(idUser,idDevice,idPufsConfig,testDir,username,password)
            else:
                print('[CLIENT-APP] - LaunchTests() - Insufficient number of devices available!')
                for device in devicesList:
                    idDevice = device['id']
                    _LockAndProgramDevice(idUser,idDevice,idPufsConfig,testDir,username,password)
        else:
            print('[CLIENT-APP] - LaunchTests() - No device available')
    else:
        print('[CLIENT-APP] - LaunchTests() - User registration failed!')
