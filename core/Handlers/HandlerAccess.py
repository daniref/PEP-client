import os

from core.Utility import *

# to use only in case of new registration!!!!!!!!!!!
def RegisterUser():
    httpUrl = urljoin(GetServerAddress(), httpGetUserData)
    params={'username': userName, 'password': password,
            'firstname' : firstName, 'lastname' : lastName,
            'email' : email, 'affiliation' : affiliation
            }
    response = requests.post(httpUrl,data=params, verify=certFile)

    # Verifica se la richiesta è andata a buon fine (codice di stato 200)
    if response.status_code == 200:
        # La risposta contiene i dispositivi disponibili, quindi possiamo accedere ai dati
        data = response.json()  # Converti la risposta in formato JSON
        if (data['result'] == True):
            print('[CLIENT-APP] - RegisterUser() - User correctly registered with id:', data['idUser'])
            return data['idUser']
        else:
            print('[CLIENT-APP] - RegisterUser() - User already registered with id:', data['idUser'])
            return data['idUser']
    else:
        # Se la richiesta non è andata a buon fine, gestisci l'errore
        print('[CLIENT-APP] - RegisterUser() - Http Post: User Registration Error:', response.status_code)
        return None

def LogInUser(username,password):
    httpUrl = urljoin(GetServerAddress(), httpLogIn)
    payload={'username': username, 'password': password}
    response = requests.post(httpUrl,data=payload,verify=certFile)

    # Verifica se la richiesta è andata a buon fine (codice di stato 200)
    if response.status_code == 200:
        data = response.json()  # Converti la risposta in formato JSON
        # La risposta contiene i dispositivi disponibili, quindi possiamo accedere ai dati
        if (data['result'] == True):
            print('[CLIENT-APP] - Login() - User correctly logged in with id: ', data['idUser'])
            return True,data['idUser']
        else:
            print('[CLIENT-APP] - Login() - User not correctly logged')
            return None,None
    else:
        # Se la richiesta non è andata a buon fine, gestisci l'errore
        print('[CLIENT-APP] - Login() - Http Post: User login error:', response.status_code)
        return None,None