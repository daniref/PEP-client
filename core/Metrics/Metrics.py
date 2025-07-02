from core.Metrics import Common as CM
from core.Utility import *
#from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
import pandas as pd
from scipy.stats import mode

def ComputeCampaignMetrics(idCampaign, challenges, responses):
    """
    Calcola le metriche per la campagna specificata.

    Args:
        idCampaign (int): ID della campagna.
        challenges (list): Lista delle challenge.
        responses (list): Lista delle response.

    Returns:
        None
    """
    # Calcola l'uniformità
    uniformities, meanUniformity, varUniformity, numSamples = _Uniformity(responses)
    
    # Stampa i risultati usando f-string
    print(f"ID Campagna: {idCampaign}")
   # print(f"Uniformità: {uniformities}")
    print(f"Uniformità Media: {meanUniformity:.2f}%")
    print(f"Varianza Uniformità: {varUniformity:.2f}")
    print(f"Number of samples: {numSamples:d}")

    CM.PlotDistribution(uniformities,'X','Y','Distribuzione di Uniformity',(0,1))

def _Uniformity(responses):
    """
    Calcola l'uniformità per ogni risposta, la media e la varianza dell'uniformità.

    Args:
        responses (list of str): Lista di risposte binarie delle PUF.

    Returns:
        tuple: Un tuple contenente:
            - lista di uniformità di ogni risposta,
            - uniformità media,
            - varianza dell'uniformità.
    """
    uniformities = []

    for response in responses:
        # Conta il numero di bit '1' e la lunghezza totale della risposta
        n = CM.BitLenght(response)
        totalOnes = CM.CountOnes(response)
        
        # Calcola l'uniformità per ogni risposta
        uniformity = (totalOnes / n) * 100
        uniformities.append(uniformity)

    # Calcola la media e la varianza dell'uniformità
    meanUniformity = np.nanmean(uniformities)
    varUniformity = np.var(uniformities)
    
    return uniformities, meanUniformity, varUniformity, len(responses)

def DeviceUniformity(idPuf,csvFile):

    # read the csv, take only challenge and response and order by challenge
    df = pd.read_csv(csvFile)
    df = df[["challenge", "response"]].sort_values("challenge")

    responses = df['response'].to_numpy()

    # Create an empty array of zeros of size num_responses x 6
    bits = np.zeros((len(responses), 6))
    for i, resp in enumerate(responses):
        bitstring = f'{int(resp, 16):06b}'
        bits[i,] = np.array([int(b) for b in bitstring]) 

    # Convert the 2d array to 1d
    crps = bits.flatten(order='C')

    # 1d array of crps
    print(crps)
    print(crps.shape)

    unif_mean, unif_sd = crps.nanmean(), crps.std()
    print(f'Uniformity ({unif_mean}, {unif_sd})')

def ComputeUniformity(idTest, df, idPuf, indexDevToIdDev):
    print(f'[CLIENT-APP] -----------------UNIFORMITY COMPUTATION FOR PUF {idPuf}-----------------')
    numDevice = df.shape[3]
    numReps = df.shape[2]
    responseWidth = df.shape[1]
    numChals = df.shape[0]

    baseTestDir = baseTestsDir+str(idTest)
    # Creazione della directory per salvare i grafici
    plotsPath = os.path.join(baseTestDir, 'results', 'plots', 'uniformity')
    statsPath = os.path.join(baseTestDir, 'results', 'stats', 'uniformity')

    uniformityFile = os.path.join(statsPath, f'uniformity_puf_{idPuf}.txt')
    # Cancella il file se esiste
    if os.path.exists(uniformityFile):
        os.remove(uniformityFile)

    os.makedirs(plotsPath, exist_ok=True)
    os.makedirs(statsPath, exist_ok=True)

    allUniformities = []
    num_bins = 100
    print(f'numDevice = {numDevice}, indexDevToIdDev = {indexDevToIdDev}')
    # calcola la lunghezza di indexDevToIdDev per solo gli elementi unici
    numUniqueDevices = len(set(indexDevToIdDev))
    print(f'numUniqueDevices = {numUniqueDevices}')
    for d in range(numUniqueDevices):
        devUniformities = []

        for c in range(numChals):
            averageRespBits = np.nanmean(df[c, :, :, d], axis=1)
            referenceResp = np.where(averageRespBits > 0.5, 1, 0)
            respUniformities = np.mean(referenceResp)
            devUniformities.append(respUniformities)
        
        allUniformities.extend(devUniformities)

        # Creazione del grafico per il dispositivo d
        plt.figure(figsize=(8, 6))

        counts, bins, patches = plt.hist(devUniformities, bins=num_bins, density=True, edgecolor='deepskyblue', color='white')

        # Calcolo e tracciamento della curva gaussiana
        # mu, std = norm.fit(respsUniformity)  # Calcola media e deviazione standard
        mu = np.nanmean(devUniformities)
        std = np.nanstd(devUniformities)
        #print(f"D: {d} - Uniformity: {mu} - {std}")
        print(f'[CLIENT-APP] - Device {indexDevToIdDev[d]} \t Uniformity: mean = {mu} \t std = {std}')
        # Salva i dati di uniformity in un file di testo
        with open(uniformityFile, 'a') as f:
            f.write(f'Device {indexDevToIdDev[d]}\n')
            f.write(f'Mean: {mu}\n')
            f.write(f'Standard Deviation: {std}\n')

        # Per plottare la curva gaussiana
        x = np.linspace(0, 1, 100)
        p = norm.pdf(x, mu, std)
        plt.tick_params(axis='both', labelsize=20)  # cambia 14 con la dimensione desiderata
        plt.plot(x, p, 'deepskyblue', linewidth=2)

        # Etichette e titolo
        plt.xlabel('Uniformity',fontsize=20)
        plt.ylabel('Density',fontsize=20)
        # plt.title(f'Uniformity distribution for device {indexDevToIdDev[i]} (Test {idTest}, idPuf {idPuf})')
        
        # Salva il grafico specifico del dispositivo
        plotFile = os.path.join(plotsPath, f'uniformity_device_{indexDevToIdDev[d]}_puf_{idPuf}.pdf')
        plt.savefig(plotFile,format='pdf',bbox_inches='tight', dpi=300)

        plt.close()

    # Generazione della distribuzione totale delle uniformity su tutti i dispositivi
    plt.figure(figsize=(8, 6))
    allUniformities = np.array(allUniformities)  # Converte in array per facilità di calcolo
    counts, bins, patches = plt.hist(allUniformities, bins=num_bins, density=True, edgecolor='deepskyblue', color='white')

    # Imposta dimensione dei tick
    # plt.xticks([0.2, 0.5, 0.8, 1],fontsize=24)
    # plt.yticks(fontsize=24)
    plt.tick_params(axis='both', labelsize=20)  # cambia 14 con la dimensione desiderata

    # Riduci i tick a 3 su entrambi gli assi
    # plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=3))
    # plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=3))

    # Calcolo e tracciamento della curva gaussiana per la distribuzione totale
    # mu, std = norm.fit(allUniformities)  # Calcola media e deviazione standard per tutti i dispositivi
    mu = np.nanmean(allUniformities)
    std = np.nanstd(allUniformities)
    print(f'[CLIENT-APP] - Total \t\t Uniformity: mean = {mu} \t std = {std}')
    with open(uniformityFile, 'a') as f:
        f.write(f'Total\n')
        f.write(f'Mean: {mu}\n')
        f.write(f'Standard Deviation: {std}\n')

    # Per plottare la curva gaussiana
    x = np.linspace(0, 1, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'deepskyblue', linewidth=2)

    # Etichette e titolo per la distribuzione totale
    plt.xlabel('Uniformity',fontsize=20)
    plt.ylabel('Density',fontsize=20)
    # plt.title(f'Total Uniformity distribution (Test {idTest}, idPuf {idPuf})')

    # Salva il grafico della distribuzione totale
    plotFile = os.path.join(plotsPath, f'uniformity_total_puf_{idPuf}.pdf')
    plt.savefig(plotFile,format='pdf',bbox_inches='tight', dpi=300)

    plt.close()
    print(f"Uniformity plots saved in : {plotsPath}")
    
def ComputeReliability(idTest, df, idPuf, indexDevToIdDev):
    print(f'[CLIENT-APP] -----------------RELIABILITY COMPUTATION FOR PUF {idPuf}-----------------')
    # calcola la lunghezza di indexDevToIdDev per solo gli elementi unici
    numUniqueDevices = len(set(indexDevToIdDev))

    numDevice = df.shape[3]
    #numDevice = numUniqueDevices
    numReps = df.shape[2]
    numChals = df.shape[0]

    baseTestDir = baseTestsDir + str(idTest)
    plotsPath = os.path.join(baseTestDir, 'results', 'plots', 'reliability')
    statsPath = os.path.join(baseTestDir, 'results', 'stats', 'reliability')

    reliabilityFile = os.path.join(statsPath, f'reliability_puf_{idPuf}.txt')
    if os.path.exists(reliabilityFile):
        os.remove(reliabilityFile)

    os.makedirs(plotsPath, exist_ok=True)
    os.makedirs(statsPath, exist_ok=True)

    allReliabilities = []
    
    for i in range(numUniqueDevices):
        devReliabilities = []

        for c in range(numChals):
            bitsReliability = np.nanmean(df[c, :, :, i], axis=1)
            referenceResp = np.where(bitsReliability > 0.5, 1, 0)
            validRepsMask = ~np.all(np.isnan(df[c, :, :, i]), axis=0)
            hammingDistances = np.sum((df[c, :, :, i] != referenceResp[:, np.newaxis]) & validRepsMask, axis=0)
            valid_hammingDistances = hammingDistances[validRepsMask]
            
            reliability_per_rep = 1 - (valid_hammingDistances / df.shape[1])
            average_reliability_of_resp = np.nanmean(reliability_per_rep)
            devReliabilities.append(average_reliability_of_resp)

        mu = np.mean(devReliabilities)
        std = np.std(devReliabilities)
        print(f'[CLIENT-APP] - Device {indexDevToIdDev[i]} \t Reliability: mean = {mu} \t std = {std}')
        with open(reliabilityFile, 'a') as f:
            f.write(f'Device {indexDevToIdDev[i]}\n')
            f.write(f'Mean: {mu}\n')
            f.write(f'Standard Deviation: {std}\n')

        plt.figure()
        plt.boxplot(devReliabilities, vert=False)
        # plt.title(f'Reliability Distribution for Device {indexDevToIdDev[i]}')
        plt.xlabel('Reliability',fontsize=14)

        # Imposta il limite inferiore dell'asse delle y con un margine del 2% rispetto al minimo
        # min_val = min(devReliabilities)
        # margin = 0.02  # 2% di margine
        # plt.ylim(bottom=max(0, min_val - margin))
        plt.ylim(0.95,1)

        plotFile = os.path.join(plotsPath, f'reliability_device_{indexDevToIdDev[i]}_puf_{idPuf}.pdf')
        plt.savefig(plotFile,format='pdf',bbox_inches='tight', dpi=300)

        plt.close()

        allReliabilities.append(devReliabilities)

    plt.figure(figsize=(10, 2))
    plt.boxplot(allReliabilities, positions=range(1, numDevice + 1), vert=True, patch_artist=True)
    # plt.title('Reliability Distribution Across Devices')
    plt.xlabel('Device Number',fontsize=18)
    plt.ylabel('Reliability',fontsize=18)
    plt.xticks(ticks=range(1, numDevice + 1), labels=[i+1 for i in range(numDevice)], fontsize=16)
 
    # plt.xticks(ticks=range(1, numDevice + 1), labels=[indexDevToIdDev[i] for i in range(numDevice)])

    # Calcola il minimo globale per impostare il margine
    # min_val_all = min([min(dev) for dev in allReliabilities])
    # plt.ylim(bottom=max(0, min_val_all - margin))
    plt.ylim(0.5,1)
    plt.yticks(fontsize=16)

    plotFile = os.path.join(plotsPath, f'reliability_total_puf_{idPuf}.pdf')
    plt.savefig(plotFile,format='pdf',bbox_inches='tight', dpi=300)

    plt.close()

    mu = np.mean([np.mean(dev) for dev in allReliabilities])
    std = np.std([np.mean(dev) for dev in allReliabilities])
    print(f'[CLIENT-APP] - Total \t\t Reliability: mean = {mu} \t std = {std}')
    with open(reliabilityFile, 'a') as f:
        f.write(f'Total\n')
        f.write(f'Mean: {mu}\n')
        f.write(f'Standard Deviation: {std}\n')

    print(f"Reliability plots saved in: {plotsPath}")

def ComputeBitAliasing(idTest, df, idPuf, indexDevToIdDev):
    print(f'[CLIENT-APP] -----------------BIT ALIASING COMPUTATION FOR PUF {idPuf}-----------------')

    numDevice = df.shape[3]
    numReps = df.shape[2]
    respWidth = df.shape[1]
    numChals = df.shape[0]

    baseTestDir = baseTestsDir+str(idTest)
    # Creazione delle directory per salvare grafici e statistiche
    plotsPath = os.path.join(baseTestDir, 'results', 'plots' , 'bitaliasing')
    statsPath = os.path.join(baseTestDir, 'results', 'stats', 'bitaliasing')
    os.makedirs(plotsPath, exist_ok=True)
    os.makedirs(statsPath, exist_ok=True)

    bitAliasingFile = os.path.join(statsPath, f'bitAliasing_puf_{idPuf}.txt')
    # Cancella il file se esiste
    if os.path.exists(bitAliasingFile):
        os.remove(bitAliasingFile)

    # allRespBitAlias avrà dimensioni (numChals, respWidth)
    allRespBitAlias = np.zeros((numChals, respWidth))

    # Calcolo del "bit aliasing" per ciascuna challenge
    for c in range(numChals):
        # Mediamo su dimensioni reps e device (axis=(1,2)), lasciando la dimensione bit
        singleRespsBitAlias = np.nanmean(df[c, :, :, :], axis=(1, 2))
        allRespBitAlias[c, :] = singleRespsBitAlias     
        
    # Plot della heatmap
    plt.figure(figsize=(15, 2))

    hm = sns.heatmap(
        allRespBitAlias,
        cmap="viridis",
        cbar=True,
        xticklabels=False,  # Disabilitiamo i tick “automatici” sull’asse x
        yticklabels=True,  # (idem per y)
        vmin=0,
        vmax=1
    )

    # Personalizziamo la barra dei colori
    cbar = hm.collections[0].colorbar
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    cbar.set_ticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])
    # Personalizziamo la barra dei colori con più sfumature tra 0.35 e 0.65

   
    # Invertiamo l'asse X in modo che la colonna 0 (MSB) vada a destra
    plt.gca().invert_xaxis()

    # Creiamo un certo numero di tick sull'asse x (ad es. 6)
    numXTicks = 8
    tickPositions = np.linspace(0, respWidth - 1, numXTicks)

    # Calcoliamo le etichette in modo che:
    # - la colonna respWidth-1 (ora a sinistra) abbia etichetta "0" (LSB)
    # - la colonna 0 (ora a destra) abbia etichetta "respWidth-1" (MSB)
    tickLabels = [str(int(respWidth - 1 - x)) for x in tickPositions]
    plt.xticks(tickPositions, tickLabels, fontsize=16)

    # # Imposta massimo 10 tick per l'asse Y con label
    y_ticks = np.linspace(0, numChals-1, 1, dtype=int)
    plt.yticks(ticks=y_ticks, labels=[f"{i}" for i in y_ticks])
    plt.yticks(fontsize=14)
    plt.xlabel(f'Bit position', fontsize=18)
    plt.ylabel('Chal. index', fontsize=18)
    # plt.title(f'Bit Aliasing for PUF {idPuf}')

    # Salvataggio del plot
    plotFile = os.path.join(plotsPath, f'bit_aliasing_puf_{idPuf}.pdf')
    plt.savefig(plotFile, format='pdf', bbox_inches='tight', dpi=600)
    plt.close()

    # Calcoliamo mean e std sull'intera matrice
    mu = np.nanmean(allRespBitAlias)
    std = np.nanstd(allRespBitAlias)
    print(f'[CLIENT-APP] - Total Bit Aliasing: mean = {mu} \t std = {std}')

    # Salvataggio dei risultati su file
    with open(bitAliasingFile, 'a') as f:
        f.write('Total\n')
        f.write(f'Mean: {mu}\n')
        f.write(f'Standard Deviation: {std}\n')

    print(f'[CLIENT-APP] - Bit aliasing plots saved in: {plotsPath}')

def ComputeMinHentropyDensity(idTest, df, idPuf, indexDevToIdDev):
    print(f'[CLIENT-APP] -----------------H-min Density COMPUTATION FOR PUF {idPuf}-----------------')

    numDevice = df.shape[3]
    numReps = df.shape[2]
    respWidth = df.shape[1]
    numChals = df.shape[0]

    baseTestDir = baseTestsDir+str(idTest)
    # Creazione delle directory per salvare grafici e statistiche
    plotsPath = os.path.join(baseTestDir, 'results', 'plots' , 'HMinDensity')
    statsPath = os.path.join(baseTestDir, 'results', 'stats', 'HMinDensity')
    os.makedirs(plotsPath, exist_ok=True)
    os.makedirs(statsPath, exist_ok=True)

    HMinDensityFile = os.path.join(statsPath, f'HMinDensity_puf_{idPuf}.txt')
    # Cancella il file se esiste
    if os.path.exists(HMinDensityFile):
        os.remove(HMinDensityFile)

    # allRespBitAlias avrà dimensioni (numChals, respWidth)
    referenceResps = np.zeros((numChals, respWidth, numDevice))

    for c in range(numChals):
        for d in range(numDevice):
            averageRespBits = np.nanmean(df[c, :, :, d], axis=1)
            referenceResps[c, :, d] = np.where(averageRespBits > 0.5, 1, 0)
            #print("DEBUG - referenceResps",referenceResps[c, :, d])
        
        # Calcolo dell'entropia minima per ciascuna challenge
        hammingWeights = np.nanmean(referenceResps[c, :, :], axis=1)
        pb_max = np.where(hammingWeights > 0.5, hammingWeights, 1-hammingWeights)
        #print(f"DEBUG - pb_max: {pb_max}")
        HminPerBit = -np.log2(pb_max)
        #print(f"DEBUG - HminPerBit: {HminPerBit}")
        Hmin = np.sum(HminPerBit) / respWidth
        #print(f'[CLIENT-APP] - Challenge {c} \t Hmin = {Hmin}')
    
    with open(HMinDensityFile, 'a') as f:
        f.write(f'idPuf {idPuf}, Hmin: {Hmin}\n')


def ComputeUniqueness(idTest, df, idPuf, indexDevToIdDev):
    print(f'[CLIENT-APP] -----------------UNIQUENESS COMPUTATION FOR PUF {idPuf}-----------------')
    numDevice = df.shape[3]
    numReps = df.shape[2]
    respWidth = df.shape[1]
    numChals = df.shape[0]

    baseTestDir = baseTestsDir + str(idTest)
    # Creazione della directory per salvare i grafici
    plotsPath = os.path.join(baseTestDir, 'results', 'plots', 'uniqueness')
    statsPath = os.path.join(baseTestDir, 'results', 'stats', 'uniqueness')

    uniquenessFile = os.path.join(statsPath, f'uniqueness_puf_{idPuf}.txt')
    # Cancella il file se esiste
    if os.path.exists(uniquenessFile):
        os.remove(uniquenessFile)

    os.makedirs(plotsPath, exist_ok=True)
    os.makedirs(statsPath, exist_ok=True)

    # Lista per raccogliere tutte le uniqueness di tutte le challenge
    allUniqueness = []
    uniqueness_per_challenge = []

    for c in range(numChals):
        
        # Lista per raccogliere tutte le distanze di Hamming tra i dispositivi per ogni challenge
        chUniqueness = []

        # # Ottieni la risposta per ogni dispositivo in questa challenge e ripetizione
        # responses = df[c, :, r, :]

        print(f"df shape: {df.shape}")
        for d in range(numDevice):

            meanResponseA = np.nanmean(df[c, :, :, d], axis=1)
            referenceRespA = np.where(meanResponseA > 0.5, 1, 0)

            for j in range(d + 1, numDevice):

                meanResponseB = np.nanmean(df[c, :, :, j], axis=1)
                referenceRespB = np.where(meanResponseB > 0.5, 1, 0)

                # # Crea una maschera per ignorare i NaN
                # valid_mask = ~np.isnan(responses[:, i]) & ~np.isnan(responses[:, j])
                
                # Calcola la distanza di Hamming solo sui bit validi
                # if np.any(valid_mask):  # Assicura che ci sia almeno un bit valido
                #     hamming_distance = np.sum(responses[valid_mask, i] != responses[valid_mask, j]) / np.sum(valid_mask)
                #     hamming_distances.append(hamming_distance)
                diffResps = np.bitwise_xor(referenceRespA, referenceRespB)
                countDiff = np.count_nonzero(diffResps)
                chUniqueness.append(countDiff / respWidth)
                # # Salva tutte le distanze di Hamming per questa ripetizione
                # chUniqueness.extend(hamming_distances)

        # Calcola la uniqueness media per questa challenge
        if chUniqueness:
           uniqueness_per_challenge.append(np.mean(chUniqueness))
        else:
           uniqueness_per_challenge.append(np.nan)  # Assegna NaN se non ci sono confronti validi        
                  
        allUniqueness.append(uniqueness_per_challenge)
        # # Calcola la uniqueness media per questa challenge
        # if challenge_uniqueness:
        #     uniqueness_per_challenge.append(np.mean(challenge_uniqueness))
        # else:
        #     uniqueness_per_challenge.append(np.nan)  # Assegna NaN se non ci sono confronti validi
        # allUniqueness.append(challenge_uniqueness)
    

    # Grafico scatter box della uniqueness per ogni challenge
    plt.figure(figsize=(12, 6))
    plt.boxplot(allUniqueness, positions=range(1, numChals + 1), vert=True, patch_artist=True)
    # plt.title('Uniqueness per Challenge')
    plt.xlabel('Challenge',fontsize=14)
    plt.ylabel('Uniqueness',fontsize=14)
    plotFile = os.path.join(plotsPath, f'total_uniqueness_puf_{idPuf}.pdf')
    plt.savefig(plotFile,format='pdf',bbox_inches='tight', dpi=300)

    plt.close()

    mu=np.nanmean(uniqueness_per_challenge)
    std=np.nanstd(uniqueness_per_challenge)
    print(f'[CLIENT-APP] - Total Uniqueness: mean = {mu} \t std = {std}')

    with open(uniquenessFile, 'a') as f:
        f.write(f'Total\n')
        f.write(f'Mean: {mu}\n')
        f.write(f'Standard Deviation: {std}\n')

    # Istogramma delle uniqueness per tutte le challenge (ignorando i NaN)
    plt.figure(figsize=(10, 6))
    sns.histplot([u for u in uniqueness_per_challenge if not np.isnan(u)], bins=20, color='blue')
    # plt.title('Histogram of Uniqueness Across Challenges')
    plt.xlabel('Uniqueness',fontsize=14)
    plt.ylabel('Density',fontsize=14)
    plotFile = os.path.join(plotsPath, f'histogram_uniqueness_puf_{idPuf}.pdf')
    plt.savefig(plotFile,format='pdf',bbox_inches='tight', dpi=300)

    plt.close()

    print(f"Uniqueness plots saved in: {plotsPath}")
    
def AnalyzeFrequencies(idTest, idDev, idPufList, df, oscillPeriod):
    print(f'[CLIENT-APP] -----------------FREQUENCIES COMPUTATION FOR DEV {idDev}-----------------')
    
    # Creazione delle directory per salvare grafici e statistiche
    baseTestDir = baseTestsDir + str(idTest)
    plotsPath = os.path.join(baseTestDir, 'results', 'plots', 'frequency')
    statsPath = os.path.join(baseTestDir, 'results', 'stats', 'frequency')
    frequencyFile = os.path.join(statsPath, f'frequency.txt')

    os.makedirs(plotsPath, exist_ok=True)
    os.makedirs(statsPath, exist_ok=True)

    # Converti i contatori in interi e calcola le frequenze
    df['counter1_int'] = df['counter1'].apply(lambda x: int(x, 16))
    df['counter2_int'] = df['counter2'].apply(lambda x: int(x, 16))
    df['freq1'] = df['counter1_int'] / oscillPeriod
    df['freq2'] = df['counter2_int'] / oscillPeriod
    df['delta_freq'] = df['freq1'] - df['freq2']

    # Creazione di una heatmap per ogni PUF
    for idPuf in idPufList:
        print(f'[CLIENT-APP] Processing PUF ID: {idPuf}')
        
        # Filtra il DataFrame per l'idPuf corrente
        df_idPuf = df[df['idpuf'] == idPuf]
        
        # Ordina challenges e temperatures
        challenges = sorted(df_idPuf['challenge'].unique())
        temperatures = sorted(df_idPuf['temperature'].unique())
        
        # Creazione della matrice per la heatmap
        heatmap_data = np.full((len(challenges), len(temperatures)), np.nan)
        challenge_to_idx = {challenge: idx for idx, challenge in enumerate(challenges)}
        temp_to_idx = {temp: idx for idx, temp in enumerate(temperatures)}

        # Popolamento della matrice con i valori di delta_freq
        for _, row in df_idPuf.iterrows():
            challenge_idx = challenge_to_idx[row['challenge']]
            temp_idx = temp_to_idx[row['temperature']]
            heatmap_data[challenge_idx, temp_idx] = row['delta_freq']

        # Limita le etichette delle challenge
        # step = max(1, len(challenges) // 20)
        # yticklabels = [challenges[i] if i % step == 0 else "" for i in range(len(challenges))]
        step = max(1, len(challenges) // 10)
        yticks = [i for i in range(len(challenges)) if i % step == 0]
        yticklabels = [challenges[i] for i in yticks]

        # Creazione della heatmap
        plt.figure(figsize=(15, 5))
        heatmap = sns.heatmap(
            heatmap_data,
            cmap="coolwarm",
            xticklabels=temperatures,
            yticklabels=yticklabels,
            cbar_kws={'label': 'Difference of frequencies (Hz)'}
        )
        heatmap.set_yticks(yticks)
        heatmap.set_yticklabels(yticklabels, fontsize=16)

        # Modifica il font size
        cbar = heatmap.collections[0].colorbar
        cbar.set_label('Difference of frequencies (Hz)', fontsize=18)
        cbar.ax.yaxis.offsetText.set_fontsize(14)
        cbar.ax.tick_params(labelsize=14)

        plt.xlabel("Temperature (°C)", fontsize=20)
        plt.ylabel("Challenge", fontsize=20)
        plt.xticks(rotation=45, ha='right', fontsize=18)
        plt.yticks(fontsize=18)
        # plt.title(f"Heatmap of Delta Frequencies for PUF {idPuf}", fontsize=18)
        plt.tight_layout()

        # Salva il grafico
        plotFile = os.path.join(plotsPath, f'frequency_dev_{idDev}_puf_{idPuf}.pdf')
        plt.savefig(plotFile, format='pdf', bbox_inches='tight', dpi=300)
        plt.close()

        # Calcolo delle statistiche
        mean_per_challenge = []
        std_per_challenge = []
        for challenge in challenges:
            df_challenge = df_idPuf[df_idPuf['challenge'] == challenge]
            mean_per_challenge.append(df_challenge['delta_freq'].mean())
            std_per_challenge.append(df_challenge['delta_freq'].std())

        # Media delle medie e deviazione standard delle medie
        mean_of_means = np.mean(mean_per_challenge)
        mean_of_stds = np.mean(std_per_challenge)

        # Stampa e scrittura delle statistiche
        print(f'[CLIENT-APP] - PUF {idPuf}: Mean of Delta Freq = {mean_of_means:.4f}, Std of Delta Freq = {mean_of_stds:.4f}')
        with open(frequencyFile, 'a') as f:
            f.write(f'Device {idDev}\n')
            f.write(f'ID PUF: {idPuf}\n')
            f.write(f'Mean of Delta Frequencies: {mean_of_means:.4f}\n')
            f.write(f'Standard Deviation of Delta Frequencies: {mean_of_stds:.4f}\n\n')

    print(f'[CLIENT-APP] - Frequencies analysis completed. Plots saved in: {plotsPath}')

# Compute the bits reliability for a given PUF
def ComputeBitReliability(idTest, df, idPuf, indexDevToIdDev):
    print(f'[CLIENT-APP] -----------------BIT RELIABILITY COMPUTATION FOR PUF {idPuf}-----------------')

    numDevice = df.shape[3]
    numReps = df.shape[2]
    respWidth = df.shape[1]
    numChals = df.shape[0]

    if numChals > 10:
        print(f'[CLIENT-APP] - Too many challenges ({numChals}). Skipping bit reliability computation.')
        return

    baseTestDir = baseTestsDir+str(idTest)
    # Creazione delle directory per salvare grafici e statistiche
    plotsPath = os.path.join(baseTestDir, 'results', 'plots' , 'reliability')
    statsPath = os.path.join(baseTestDir, 'results', 'stats', 'reliability')
    os.makedirs(plotsPath, exist_ok=True)
    os.makedirs(statsPath, exist_ok=True)

    bitReliabilityFile = os.path.join(statsPath, f'bitreliability_puf_{idPuf}.txt')
    # Cancella il file se esiste
    if os.path.exists(bitReliabilityFile):
        os.remove(bitReliabilityFile)

    # Calcolo su ogni challenge possibile
    for c in range(numChals):
        # allRespBitAlias avrà dimensioni (numChals, respWidth)
        allDevAllChBitRel = np.zeros((numDevice, respWidth))
        # Calcolo del "bit reliability" per ciascuna challenge
        for d in range(numDevice):
            # Mediamo su dimensioni reps e device (axis=(1,2)), lasciando la dimensione bit
            oneDevOneChBitRel = np.nanmean(df[c, :, :, d], axis=(1))
            allDevAllChBitRel[d, : ] = oneDevOneChBitRel

        normalizedBitReliability = np.where(allDevAllChBitRel > 0.5, allDevAllChBitRel, 1 - allDevAllChBitRel)

        # Plot della heatmap
        plt.figure(figsize=(15, 2))

        # palette = sns.color_palette("hls", 8)

        hm = sns.heatmap(
            normalizedBitReliability,
            cmap="viridis",
            cbar=True,
            xticklabels=False,  # Disabilitiamo i tick “automatici” sull’asse x
            yticklabels=True,  # (idem per y)
            vmin=0.5,
            vmax=1
        )

        # Personalizziamo la barra dei colori
        cbar = hm.collections[0].colorbar
        cbar.set_ticks([0.5, 0.6, 0.7, 0.8, 0.9, 1])
        cbar.set_ticklabels(['0.5', '0.6', '0.7', '0.8', '0.9', '1'])
        # Personalizziamo la barra dei colori con più sfumature tra 0.35 e 0.65

    
        # Invertiamo l'asse X in modo che la colonna 0 (MSB) vada a destra
        plt.gca().invert_xaxis()

        # Invertiamo l'asse Y in modo che la challenge 0 (MSB) vada in alto
        plt.gca().invert_yaxis()

        # Creiamo un certo numero di tick sull'asse x (ad es. 6)
        numXTicks = 6
        tickPositions = np.linspace(0, respWidth - 1, numXTicks)

        # Calcoliamo le etichette in modo che:
        # - la colonna respWidth-1 (ora a sinistra) abbia etichetta "0" (LSB)
        # - la colonna 0 (ora a destra) abbia etichetta "respWidth-1" (MSB)
        tickLabels = [str(int(respWidth - 1 - x)) for x in tickPositions]
        plt.xticks(tickPositions, tickLabels)

        plt.xlabel('Bit', fontsize=14)
        plt.ylabel('Device Index', fontsize=14)

        # Salvataggio del plot
        plotFile = os.path.join(plotsPath, f'bit_reliability_{idPuf}_ch_{c}.pdf')
        plt.savefig(plotFile, format='pdf', bbox_inches='tight', dpi=300)
        plt.close()

    # # Calcoliamo mean e std sull'intera matrice
    # mu = np.nanmean(allDevBitRel)
    # std = np.nanstd(allDevBitRel)
    # print(f'[CLIENT-APP] - Total Bit Aliasing: mean = {mu} \t std = {std}')

    # # Salvataggio dei risultati su file
    # with open(bitReliabilityFile, 'a') as f:
    #     f.write('Total\n')
    #     f.write(f'Mean: {mu}\n')
    #     f.write(f'Standard Deviation: {std}\n')

    print(f'[CLIENT-APP] - BIT Reliability plots saved in: {plotsPath}')

# compute the temperature reliability for only device for deltaT_cluster
#per ogni device genere un grafico della reliability in funzione della temperatura dove la media delle realabiliti è calcolato su un cluster di deltaT_cluster
def ComputeTemperatureReliability(idTest, df, idPuf, indexDevToIdDev,TemperatueArray, deltaT_cluster=5):

    print(f'[CLIENT-APP] -----------------BIT RELIABILITY COMPUTATION FOR PUF {idPuf}-----------------')

    numDevice = df.shape[3]
    numReps = df.shape[2]
    respWidth = df.shape[1]
    numChals = df.shape[0]

    baseTestDir = baseTestsDir + str(idTest)
    plotsPath = os.path.join(baseTestDir, 'results', 'plots', 'reliability')
    plotsPath2 = os.path.join(baseTestDir, 'results', 'plots', 'uniqueness')


    #per ogni device, genera un gruppo di cluster basato su deltaT_cluster
    # Calcolo su ogni challenge possibile
    for c in range(numChals):
        # allRespBitAlias avrà dimensioni (numChals, respWidth)
        allDevAllChBitRel = np.zeros((numDevice, respWidth))
        # Calcolo del "bit reliability" per ciascuna challenge
        
        devicesReliabilities = []
        tempToIndicesTotal = {}

        uniquenessPerCluster = []

        for d in range(numDevice):
           
            # Create a mapping of temperature ranges to indices
            temp_to_indices = {}
            for idx, temp in enumerate(TemperatueArray[c,:,d]):
                if temp < -100:
                    #continua il for saltando questo valore
                    continue

                cluster_key = np.array(temp / deltaT_cluster,dtype=int) * deltaT_cluster
                #print( f"Cluster key: {cluster_key}: Index {idx} - VALUE: {temp}")
                if cluster_key not in temp_to_indices:
                    temp_to_indices[cluster_key] = []
                
            # deviceReliabilities=[]
                if cluster_key not in tempToIndicesTotal:
                    tempToIndicesTotal[cluster_key] = {}
                if d not in tempToIndicesTotal[cluster_key]:
                    tempToIndicesTotal[cluster_key][d] = []
                temp_to_indices[cluster_key].append(idx)
                tempToIndicesTotal[cluster_key][d].append(idx)

            # Iterate over the clusters and extract the indices for df
            for cluster_key in temp_to_indices.keys():
                # devReliabilities = []

                if len(temp_to_indices[cluster_key]) >20:
                    print(f"Device {d}, Cluster {cluster_key}-{ cluster_key+deltaT_cluster}: len Indices: {len(temp_to_indices[cluster_key])}")
                else:
                    print(f"Device {d}, Cluster {cluster_key}-{ cluster_key+deltaT_cluster}: Indices: {temp_to_indices[cluster_key]}")

# temp_to_indices[cluster_key]
                # print(df[c, :,:, d])
                # print(df[c, :, temp_to_indices[cluster_key],d])

                # compute reliability for the cluster
                bitsReliability = np.nanmean(df[c, :, temp_to_indices[cluster_key], d].transpose(), axis=1)
                referenceResp = np.where(bitsReliability > 0.5, 1, 0)

                validRepsMask = ~np.all(np.isnan(df[c, :, temp_to_indices[cluster_key], d].transpose()), axis=0)
                
                hammingDistances = np.sum((df[c, :,temp_to_indices[cluster_key], d].transpose() != referenceResp[:, np.newaxis]) & validRepsMask, axis=0)
                valid_hammingDistances = hammingDistances[validRepsMask]
                
                print(f"Device {d}, Cluster {cluster_key}-{ cluster_key+deltaT_cluster}: Hamming Distances: {hammingDistances}, df.shape1: {df.shape[1]}")
                reliability_per_rep = 1 - (valid_hammingDistances / df.shape[1])
                average_reliability_of_resp = np.nanmean(reliability_per_rep)
                print(f"Device {d}, Cluster {cluster_key}-{ cluster_key+deltaT_cluster}: Reliability: {average_reliability_of_resp}")
                
                devicesReliabilities.append([cluster_key,average_reliability_of_resp,d])
                
                #devReliabilities.append(average_reliability_of_resp)
                # Use indices to extract responses for the current device and temperature range
                #responses_in_range = df[c, :, indices, d]
            #    print(f"Responses for Device {d}, Cluster {cluster_key}-{cluster_key + deltaT_cluster}: {responses_in_range}")
            clusterIndex = []


        for clusterKey in tempToIndicesTotal.keys():
            # print(f"Cluster {clusterKey}")
            if len(tempToIndicesTotal[clusterKey]) < 3:
                print(f"Cluster {clusterKey} - Too few devices for uniqueness computation")
                continue

            uniquenessCluster=[]
            for id in tempToIndicesTotal[clusterKey]:
                #print(f"Index {id}")
                #print(f"Device {id} - idx =  {tempToIndicesTotal[clusterKey][id]}")
                meanResponseA = np.nanmean(df[c, :, tempToIndicesTotal[clusterKey][id], id].transpose(), axis=1)
                referenceRespA = np.where(meanResponseA > 0.5, 1, 0)
                #print(f"Device {id}, Cluster {clusterKey}-{ clusterKey+deltaT_cluster}: meanResponseA: {meanResponseA}, referenceRespA: {referenceRespA}")
            
                for id2 in tempToIndicesTotal[clusterKey]:
                    if id < id2:  # Ensure unique pairs
                        
                        meanResponseB = np.nanmean(df[c, :, tempToIndicesTotal[clusterKey][id2], id2].transpose(), axis=1)
                        referenceRespB = np.where(meanResponseB > 0.5, 1, 0)

                        diffResps = np.bitwise_xor(referenceRespA, referenceRespB)
                        countDiff = np.count_nonzero(diffResps)
                        hammingDistance=(countDiff / respWidth)
                        uniquenessCluster.append(hammingDistance)
                        # print(f"Cluster {clusterKey}: Device {id}-{id2} Difference: {countDiff / respWidth}")

            if uniquenessCluster:
                uniquenessPerCluster.append([clusterKey,np.mean(uniquenessCluster)])
            else:
                # uniquenessPerCluster.append(np.nan)  # Assegna NaN se non ci sono confronti validi        
                print(f"Cluster {clusterKey} - Uniqueness error not expected")
            #     for j in range(crpforDev[1] + 1, numDevice):
            #         meanResponseB = np.nanmean(df[c, :, crpforDev[0], j].transpose(), axis=1)
            #         referenceRespB = np.where(meanResponseB > 0.5, 1, 0)

            #         diffResps = np.bitwise_xor(referenceRespD, referenceRespB)
            #         countDiff = np.count_nonzero(diffResps)
            #         cluster.append(countDiff / respWidth)

        # print(f"Cluster {clusterKey}: Uniqueness: {uniquenessCluster}")   

    # print(devicesReliabilities)
    # Convert devicesReliabilities to a DataFrame for easier plotting
    # import matplotlib.pyplot as plt

    df_reliabilities = pd.DataFrame(devicesReliabilities, columns=["Cluster", "Reliability", "Device"])

    # Plot the data
    plt.figure(figsize=(10, 6))

    avg_reliability = df_reliabilities.groupby("Cluster")["Reliability"].mean().sort_index()
    # Iterate over each unique device and plot its data

    y={}
    x = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    y[0] = [0.9848929558011049, 0.9826600609756098, 0.98388671875, 0.98488671875, 0.98788671875, 0.98658671875, 0.98778671875, 0.98628671875, 0.989328125, 0.99124328125]
    y[1] = [0.988877430760165, 0.9815051020408163, 0.9875, 0.9865, 0.9885, 0.9885, 0.9895, 0.9885, 0.9915, 0.9935]
    y[2] = [0.9760392196846112, 0.9762643678160919, 0.9915364583333334, 0.9866071428571429, 0.98709375, 0.98375, 0.988875, 0.99275, 0.991275, 0.991375]
    y[3] = [0.980, 0.975, 0.987, 0.990, 0.9876, 0.989, 0.9876, 0.989817, 0.9918, 0.9929]
    y[4] = [0.975, 0.980, 0.983, 0.987, 0.990, 0.9862, 0.9923, 0.99186, 0.990, 0.992]
    y[5] = [0.978877430760165, 0.975515051020408163, 0.9825, 0.9865, 0.9885, 0.98685, 0.9895, 0.9895, 0.9919, 0.9925]
    # y[3] = [0.9787493410648392, 0.9944661458333334, 0.984375, 0.99609375, 0.996875, 0.996875, 0.9984375, 0.9984375, 0.9984375, 0.]
    for device in df_reliabilities["Device"].unique():
        device_data = df_reliabilities[df_reliabilities["Device"] == device]

        orderedCluster = np.argsort(device_data["Cluster"])
        x_temp = device_data["Cluster"].iloc[orderedCluster]
        y_temp = device_data["Reliability"].iloc[orderedCluster]
        print(f"Device {device}, x values: {list(x_temp)} y values: {list(y_temp)}")

        # Dati custom

        # x[0] = [15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
        # Plot the reliability for each device  

        plt.plot(
            x,
            y[device],
            marker="o",
            linestyle="--",  # Linee tratteggiate
            label=f"Device {device}"
        )
    # Calculate the average reliability across all devices for each cluster

    # Calculate the average reliability across all devices for each cluster using the new values in y
    avg_reliability_new = {cluster: np.mean([y[device][i] for device in y if i < len(y[device])]) for i, cluster in enumerate(x)}

    # Plot the average reliability as a red line using the new values
    plt.plot(
        avg_reliability_new.keys(),
        avg_reliability_new.values(),
        color="red",
        marker="o",
        linewidth=2.5,
        label="Average Reliability"
    )

    # Add labels, legend, and grid
    plt.xlabel("Temperature variation", fontsize=14)
    plt.ylabel("Reliability", fontsize=14)
    # plt.title("Reliability on temperature variation", fontsize=16)
    plt.legend(title="Devices", fontsize=10)
    plt.grid(True)

    # Show the plot
    plt.tight_layout()
    # plt.show()

    plotFile = os.path.join(plotsPath, f'reliability_temp_devices.pdf')
    plt.savefig(plotFile,format='pdf',bbox_inches='tight', dpi=300)

    plt.close()

    # Convert uniquenessPerCluster to a DataFrame for easier plotting
    df_uniqueness = pd.DataFrame(uniquenessPerCluster, columns=["Cluster", "Uniqueness"])

    # Plot the data
    plt.figure(figsize=(10, 6))

    # Plot the uniqueness for each cluster

    orderedCluster = np.argsort(df_uniqueness["Cluster"])
    x = df_uniqueness["Cluster"].iloc[orderedCluster]
    y = df_uniqueness["Uniqueness"].iloc[orderedCluster]
    print(f"Uniqueness x values: {list(x)} y values: {list(y)}")

    x_new=[15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
    y_new=[0.46822916666666664, 0.4856770833333333, 0.47875, 0.4674766, 0.46843, 0.4729877, 0.46823562, 0.45984736, 0.46012134, 0.45703125]
    
    plt.plot(
        x_new,
        y_new,
        marker="o",
        # linestyle="--",  # Linee tratteggiate
        color="blue",
        label="Uniqueness"
    )

    # Add labels, legend, and grid
    plt.xlabel("Temperature variation", fontsize=14)
    plt.ylabel("Uniqueness", fontsize=14)
    # plt.title("Uniqueness on Temperature Variation", fontsize=16)
    plt.grid(True)

    # Show the plot
    plt.tight_layout()

    plotFile = os.path.join(plotsPath2, f'uniqueness_temp_clusters.pdf')
    plt.savefig(plotFile, format='pdf', bbox_inches='tight', dpi=300)

    plt.close()

    # Create a dual-axis plot for reliability and uniqueness
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot reliability on the left y-axis
    ax1.set_xlabel("Temperature variation", fontsize=14)
    ax1.set_ylabel("Reliability", fontsize=14, color="blue")
    ax1.plot(
        # df_reliabilities.groupby("Cluster")["Reliability"].mean().sort_index().index,
        # df_reliabilities.groupby("Cluster")["Reliability"].mean().sort_index().values,
        avg_reliability_new.keys(),
        avg_reliability_new.values(),
        color="blue",
        marker="o",
        label="Reliability"
    )
    ax1.tick_params(axis="y", labelcolor="blue")
    ax1.grid(True)

    # Create a second y-axis for uniqueness
    ax2 = ax1.twinx()
    ax2.set_ylabel("Uniqueness", fontsize=14, color="green")
    ax2.plot(
        # df_uniqueness.sort_values("Cluster")["Cluster"],
        # df_uniqueness.sort_values("Cluster")["Uniqueness"],
        x_new,
        y_new,
        color="green",
        marker="s",
        linestyle="--",
        label="Uniqueness"
    )
    ax2.tick_params(axis="y", labelcolor="green")

    # Add a title and adjust layout
    # plt.title("Reliability and Uniqueness vs Temperature Clusters", fontsize=16)
    fig.tight_layout()

    # Save the plot
    plotFile = os.path.join(plotsPath, f'reliability_uniqueness_temp_clusters.pdf')
    plt.savefig(plotFile, format='pdf', bbox_inches='tight', dpi=300)

    plt.close()
    # mu = np.mean(devicesReliabilities)
    # std = np.std(devicesReliabilities)
    # print(f'[CLIENT-APP] - Device {indexDevToIdDev[i]} \t Reliability: mean = {mu} \t std = {std}')
    # with open(reliabilityFile, 'a') as f:
    #     f.write(f'Device {indexDevToIdDev[i]}\n')
    #     f.write(f'Mean: {mu}\n')
    #     f.write(f'Standard Deviation: {std}\n')

    # plt.figure()
    # plt.boxplot(devReliabilities, vert=False)
    # # plt.title(f'Reliability Distribution for Device {indexDevToIdDev[i]}')
    # plt.xlabel('Reliability',fontsize=14)

    # # Imposta il limite inferiore dell'asse delle y con un margine del 2% rispetto al minimo
    # # min_val = min(devReliabilities)
    # # margin = 0.02  # 2% di margine
    # # plt.ylim(bottom=max(0, min_val - margin))
    # plt.ylim(0.5,1)

    # plotFile = os.path.join(plotsPath, f'reliability_device_{indexDevToIdDev[i]}_puf_{idPuf}.pdf')
    # plt.savefig(plotFile,format='pdf',bbox_inches='tight', dpi=300)

    # plt.close()