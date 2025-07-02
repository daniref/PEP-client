from core.Utility import *

POWER_UP = 'up'
POWER_DOWN = 'down'

def PowerUp(item,username,password):
    httpUrl = urljoin(GetServerAddress(), httpGetPower)
    data={'item':item,'state': POWER_UP,'username': username, 'password': password}
    response = requests.post(httpUrl,data=data, verify=certFile)

    # Verifica se la richiesta è andata a buon fine (codice di stato 200)
    if response.status_code == 200:
        try:
            # Ottieni la risposta JSON
            data = response.json()
            
            # Estrai il parametro 'result' dalla risposta JSON
            result = data.get('result')
            return result

        except ValueError:
            # Gestione dell'errore nel caso la risposta non sia in formato JSON
            print('[CLIENT-APP] - Powering request- Unable to parse JSON response.')

    else:
        # Se la richiesta non è andata a buon fine, gestisci l'errore
        print('[CLIENT-APP] - PowerUpBoards() - Request to power up failed with error ', response.status_code)
        return False
    
def PowerDown(item,username,password):
    httpUrl = urljoin(GetServerAddress(), httpGetPower)
    data={'item':item,'state': POWER_DOWN,'username': username, 'password': password}
    response = requests.post(httpUrl,data=data, verify=certFile)

    # Verifica se la richiesta è andata a buon fine (codice di stato 200)
    if response.status_code == 200:
        try:
            # Ottieni la risposta JSON
            data = response.json()
            
            # Estrai il parametro 'result' dalla risposta JSON
            result = data.get('result')
            return result

        except ValueError:
            # Gestione dell'errore nel caso la risposta non sia in formato JSON
            print('[CLIENT-APP] - Powering request- Unable to parse JSON response.')

    else:
        # Se la richiesta non è andata a buon fine, gestisci l'errore
        print('[CLIENT-APP] - PowerDownBoardsBoards() - Request to power down failed with error ', response.status_code)
        return False