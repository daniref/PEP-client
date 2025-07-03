import os

from core.Utility import *

# To use only in case of new registration!!!!!!!!!!!
def RegisterUser():
    """
    Registers a new user by sending a POST request to the server with user details.
    Returns:
        str: The user ID if registration is successful, or None if there was an error.
    """
    httpUrl = urljoin(GetServerAddress(), httpGetUserData)
    params={'username': userName, 'password': password,
            'firstname' : firstName, 'lastname' : lastName,
            'email' : email, 'affiliation' : affiliation
            }
    response = requests.post(httpUrl,data=params, verify=certFile)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # The answer contains the available devices, so we can access the data
        data = response.json()  # Convert the response to JSON format
        if (data['result'] == True):
            print('[CLIENT-APP] - RegisterUser() - User correctly registered with id:', data['idUser'])
            return data['idUser']
        else:
            print('[CLIENT-APP] - RegisterUser() - User already registered with id:', data['idUser'])
            return data['idUser']
    else:
        # If the request was not successful, handle the error
        print('[CLIENT-APP] - RegisterUser() - Http Post: User Registration Error:', response.status_code)
        return None

def LogInUser(username,password):
    """
    Logs in a user by sending a POST request to the server with the username and password.
    Args:
        username (str): The username of the user.
        password (str): The password of the user.
    Returns:
        tuple: A tuple containing a boolean indicating success and the user ID if successful, or None if not.
    """
    httpUrl = urljoin(GetServerAddress(), httpLogIn)
    payload={'username': username, 'password': password}
    response = requests.post(httpUrl,data=payload,verify=certFile)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        data = response.json()  # Convert the response to JSON format  
        # The answer contains the available devices, so we can access the data
        if (data['result'] == True):
            print('[CLIENT-APP] - Login() - User correctly logged in with id: ', data['idUser'])
            return True,data['idUser']
        else:
            print('[CLIENT-APP] - Login() - User not correctly logged')
            return None,None
    else:
        # The request was not successful, handle the error
        print('[CLIENT-APP] - Login() - Http Post: User login error:', response.status_code)
        return None,None