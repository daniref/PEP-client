from core.Utility import *

POWER_UP = 'up'
POWER_DOWN = 'down'

def PowerUp(item,username,password):
    """
    Powers up a board/fan by sending a POST request to the server with the item and state.
    Args:
        item (str): The item to power up.
        username (str): The username of the user.
        password (str): The password of the user.
    Returns:
        bool: True if the power up request was successful, False otherwise.
    """
    httpUrl = urljoin(GetServerAddress(), httpGetPower)
    data={'item':item,'state': POWER_UP,'username': username, 'password': password}
    response = requests.post(httpUrl,data=data, verify=certFile)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        try:
            # Get the JSON response
            data = response.json()
            # Extract the 'result' parameter from the JSON response
            result = data.get('result')
            return result

        except ValueError:
            # Handle the error if the response is not in JSON format
            print('[CLIENT-APP] - Powering request- Unable to parse JSON response.')

    else:
        # If the request was not successful, handle the error
        print('[CLIENT-APP] - PowerUpBoards() - Request to power up failed with error ', response.status_code)
        return False
    
def PowerDown(item,username,password):
    """
    Powers down a board/fan by sending a POST request to the server with the item and state.
    Args:
        item (str): The item to power down.
        username (str): The username of the user.
        password (str): The password of the user.
    Returns:
        bool: True if the power down request was successful, False otherwise.
    """
    httpUrl = urljoin(GetServerAddress(), httpGetPower)
    data={'item':item,'state': POWER_DOWN,'username': username, 'password': password}
    response = requests.post(httpUrl,data=data, verify=certFile)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        try:
            # Get the JSON response 
            data = response.json()
            
            # Extract the 'result' parameter from the JSON response
            result = data.get('result')
            return result

        except ValueError:
            # Handle the error if the response is not in JSON format
            print('[CLIENT-APP] - Powering request- Unable to parse JSON response.')

    else:
        # If the request was not successful, handle the error
        print('[CLIENT-APP] - PowerDownBoardsBoards() - Request to power down failed with error ', response.status_code)
        return False