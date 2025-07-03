import os

from core.Utility import *
import core.Handlers.HandlerAccess as HA

def _GetDevicesAvailable(username,password):
    """
    Retrieves the list of available devices from the server.
    Args:
        username (str): The username of the user.
        password (str): The password of the user.
    Returns:
        list: A list of available devices if the request is successful, None otherwise.
    """
    data = {
        'username':username,
        'password':password
    }
    httpUrl = urljoin(GetServerAddress(), httpGetDevAv)
    # Execute a POST request to the server
    response = requests.post(httpUrl, data=data, verify=certFile)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # The answer contains the available devices, so we can access the data
        devices = response.json()  # Convert the response to JSON format
        return devices
    else:
        # if the request was not successful, handle the error
        print('[CLIENT-APP] - Http Get: Devs Available Error:', response.status_code)

def _LockDeviceByIdDev(idUser,idDevice,username,password):
    """
    Locks a device by its ID and sets its state to unavailable.
    Args:
        idUser (str): The ID of the user.
        idDevice (str): The ID of the device to lock.
        username (str): The username of the user.
        password (str): The password of the user.
    Returns:
        dict: The result of the lock operation if successful, None otherwise.
    """
    httpUrl = urljoin(GetServerAddress(), httpGetDevAv)
    params={'idUser': idUser, 'idDevice': idDevice, 'state': 'unavailable','username':username,'password':password}
    response = requests.post(httpUrl,data=params, verify=certFile)

    # check if the request was successful (status code 200)
    if response.status_code == 200:
        # The answer contains the result of the lock operation, so we can access the data
        resultOperation = response.json()  # Convert the response to JSON format
        return resultOperation
    else:
        # If the request was not successful, handle the error
        print('[CLIENT-APP] - Http Get: Lock Device Error:', response.status_code)

def _SendsPufsConfig(idUser,testDir,username,password):
    """
    Sends the PUF configuration and bitstream files to the server.
    Args:
        idUser (str): The ID of the user.
        testDir (str): The directory where the test files are located.
        username (str): The username of the user.
        password (str): The password of the user.
    Returns:
        tuple: A tuple containing a boolean indicating success and the ID of the PUF configuration if successful, None otherwise.
    """
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
    """
    Sends the experimental configuration file to the server for a specific device.
    Args:
        idUser (str): The ID of the user.
        idDevice (str): The ID of the device.
        idPufsConfig (str): The ID of the PUF configuration.
        testDir (str): The directory where the test files are located.
        username (str): The username of the user.
        password (str): The password of the user.
    Returns:
        tuple: A tuple containing a boolean indicating success and the ID of the campaign if successful, None otherwise.
    """
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
    """
    Locks a device by its ID and programs it with the PUF configuration and experimental configuration.
    Args:
        idUser (str): The ID of the user.
        idDevice (str): The ID of the device to lock and program.
        idPufsConfig (str): The ID of the PUF configuration.
        testDir (str): The directory where the test files are located.
        username (str): The username of the user.
        password (str): The password of the user.
    """
    resOp = _LockDeviceByIdDev(idUser,idDevice,username,password) #lock the first device 
    if resOp:
        print('[CLIENT-APP] - LockAndProgramDevice() - Device with id ',idDevice,' locked successfully!')

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
    """
    Saves the campaign data to a CSV file in the specified test directory.
    Args:
        idCampaign (str): The ID of the campaign.
        idDevice (str): The ID of the device.
        testDirectory (str): The directory where the test files are located.
    """
    # check if the test directory exists, if not create it
    if not os.path.exists(testDirectory):
        os.makedirs(testDirectory)
    
    # Path of the CSV file where the campaign data will be saved
    filePath = os.path.join(testDirectory, 'campaigns.csv')
    
    # if the file exists, so we write the header only once
    fileExists = os.path.isfile(filePath)
    
    # Open the file in append mode to add new data
    with open(filePath, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # if the file does not exist, write the header
        if not fileExists:
            writer.writerow(['idCampaign', 'idDev'])
        
        # write the campaign data
        writer.writerow([idCampaign, idDevice])

def LaunchTests(numDevices,idTest,idUser,username,password):
    """
    Launches tests by locking and programming devices with the specified PUF configuration and experimental configuration.
    Args:
        numDevices (int): The number of devices to use for the tests.
        idTest (int): The ID of the test.
        idUser (str): The ID of the user.
        username (str): The username of the user.
        password (str): The password of the user.
    """
    testDir = baseTestsDir+str(idTest)+'/'
    print('[CLIENT-APP] - LaunchTests() - Request to program ', numDevices, 'devices!')
    # user not registered
    if idUser:
        print('[CLIENT-APP] - LaunchTests() - User registered with id: ', idUser)

        devicesList = _GetDevicesAvailable(username,password) #ask the server for available devices
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
