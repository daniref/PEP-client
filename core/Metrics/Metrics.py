from core.Metrics import Common as CM
from core.Utility import *
from matplotlib.ticker import MaxNLocator
import pandas as pd
from scipy.stats import mode

def ComputeUniformity(idTest, df, idPuf, indexDevToIdDev):
    """
    Computes the uniformity of responses for a given PUF (Physically Unclonable Function) 
    across multiple devices and challenges, and generates corresponding plots and statistics.
    Parameters:
        idTest (int): Identifier for the test.
        df (numpy.ndarray): A 4-dimensional array containing response data with dimensions 
            [numChals, responseWidth, numReps, numDevice].
        idPuf (int): Identifier for the PUF being analyzed.
        indexDevToIdDev (list): A list mapping device indices to their unique identifiers.
    Returns:
        None
    Side Effects:
        - Creates directories to save plots and statistics if they do not exist.
        - Saves uniformity statistics (mean and standard deviation) for each device and overall 
          in a text file.
        - Generates and saves histograms and Gaussian curve plots for uniformity distribution 
          for each device and overall.
    Notes:
        - Uniformity is computed as the mean of the binary responses for each challenge.
        - Gaussian curves are fitted to the uniformity distributions for visualization.
        - The function handles NaN values in the data by using `np.nanmean` and `np.nanstd`.
    Example:
        ComputeUniformity(
            idTest=1,
            df=response_data,
            idPuf=42,
            indexDevToIdDev=[0, 1, 2, 3]
        )
    """

    print(f'[CLIENT-APP] -----------------UNIFORMITY COMPUTATION FOR PUF {idPuf}-----------------')
    numDevice = df.shape[3]
    numReps = df.shape[2]
    responseWidth = df.shape[1]
    numChals = df.shape[0]

    baseTestDir = baseTestsDir+str(idTest)
    # Creation of directories to save plots and statistics
    plotsPath = os.path.join(baseTestDir, 'results', 'plots', 'uniformity')
    statsPath = os.path.join(baseTestDir, 'results', 'stats', 'uniformity')

    uniformityFile = os.path.join(statsPath, f'uniformity_puf_{idPuf}.txt')
    # Remove the file if it exists
    if os.path.exists(uniformityFile):
        os.remove(uniformityFile)

    os.makedirs(plotsPath, exist_ok=True)
    os.makedirs(statsPath, exist_ok=True)

    allUniformities = []
    num_bins = 100
    print(f'numDevice = {numDevice}, indexDevToIdDev = {indexDevToIdDev}')
    # Compute the number of unique devices
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

        # Creation of the histogram for the current device
        plt.figure(figsize=(8, 6))

        counts, bins, patches = plt.hist(devUniformities, bins=num_bins, density=True, edgecolor='deepskyblue', color='white')

        # Compute the Gaussian curve for the current device
        # mu, std = norm.fit(respsUniformity)  # Computes mean and standard deviation for the current device
        mu = np.nanmean(devUniformities)
        std = np.nanstd(devUniformities)
        #print(f"D: {d} - Uniformity: {mu} - {std}")
        print(f'[CLIENT-APP] - Device {indexDevToIdDev[d]} \t Uniformity: mean = {mu} \t std = {std}')
        # Save the statistics for the current device
        with open(uniformityFile, 'a') as f:
            f.write(f'Device {indexDevToIdDev[d]}\n')
            f.write(f'Mean: {mu}\n')
            f.write(f'Standard Deviation: {std}\n')

        # Plot the Gaussian curve for the current device
        x = np.linspace(0, 1, 100)
        p = norm.pdf(x, mu, std)
        plt.tick_params(axis='both', labelsize=20)
        plt.plot(x, p, 'deepskyblue', linewidth=2)

        # Labels and title
        plt.xlabel('Uniformity',fontsize=20)
        plt.ylabel('Density',fontsize=20)
        # plt.title(f'Uniformity distribution for device {indexDevToIdDev[i]} (Test {idTest}, idPuf {idPuf})')
        
        # Save the plot for the current device
        plotFile = os.path.join(plotsPath, f'uniformity_device_{indexDevToIdDev[d]}_puf_{idPuf}.pdf')
        plt.savefig(plotFile,format='pdf',bbox_inches='tight', dpi=300)

        plt.close()

    # Generatione of the histogram for the total uniformity across all devices
    plt.figure(figsize=(8, 6))
    allUniformities = np.array(allUniformities)  # Convert to numpy array for consistency
    counts, bins, patches = plt.hist(allUniformities, bins=num_bins, density=True, edgecolor='deepskyblue', color='white')

    # Set the x-axis limits to [0, 1]
    # plt.xticks([0.2, 0.5, 0.8, 1],fontsize=24)
    # plt.yticks(fontsize=24)
    plt.tick_params(axis='both', labelsize=20)  # cambia 14 con la dimensione desiderata

    # Reduce the number of ticks on both axes
    # plt.gca().xaxis.set_major_locator(MaxNLocator(nbins=3))
    # plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=3))

    # Compute the Gaussian curve for the total uniformity
    # mu, std = norm.fit(allUniformities)  # Computes mean and standard deviation for the total uniformity
    mu = np.nanmean(allUniformities)
    std = np.nanstd(allUniformities)
    print(f'[CLIENT-APP] - Total \t\t Uniformity: mean = {mu} \t std = {std}')
    with open(uniformityFile, 'a') as f:
        f.write(f'Total\n')
        f.write(f'Mean: {mu}\n')
        f.write(f'Standard Deviation: {std}\n')

    # Plot the Gaussian curve for the total uniformity
    x = np.linspace(0, 1, 100)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, 'deepskyblue', linewidth=2)

    # Labels and title
    plt.xlabel('Uniformity',fontsize=20)
    plt.ylabel('Density',fontsize=20)
    # plt.title(f'Total Uniformity distribution (Test {idTest}, idPuf {idPuf})')

    # Save the plot for the total uniformity
    plotFile = os.path.join(plotsPath, f'uniformity_total_puf_{idPuf}.pdf')
    plt.savefig(plotFile,format='pdf',bbox_inches='tight', dpi=300)

    plt.close()
    print(f"Uniformity plots saved in : {plotsPath}")
    
def ComputeReliability(idTest, df, idPuf, indexDevToIdDev):
    """
    Computes the reliability of a Physical Unclonable Function (PUF) across multiple devices 
    and challenges, and generates statistical reports and plots.
    Parameters:
    -----------
    idTest : int
        Identifier for the test being performed.
    df : numpy.ndarray
        A 4-dimensional array containing the response data. The dimensions are expected to be:
        [numChals, numBits, numReps, numDevices].
    idPuf : int
        Identifier for the PUF being analyzed.
    indexDevToIdDev : list
        A list mapping device indices to their unique identifiers.
    Outputs:
    --------
    - Reliability statistics (mean and standard deviation) for each device and overall.
    - Boxplot visualizations of reliability distributions for each device and across all devices.
    - Reliability statistics and plots are saved to the appropriate directories.
    Notes:
    ------
    - The function calculates reliability as the proportion of consistent responses across 
      repetitions for each challenge.
    - Reliability statistics are saved in a text file, and plots are saved as PDF files.
    - The function ensures that directories for saving results are created if they do not exist.
    Raises:
    -------
    - FileNotFoundError: If the base test directory does not exist.
    - ValueError: If the input data dimensions are inconsistent with the expected format.
    Example:
    --------
    ComputeReliability(
        idTest=1, 
        df=response_data, 
        idPuf=42, 
        indexDevToIdDev=[101, 102, 103]
    )
    """
    
    print(f'[CLIENT-APP] -----------------RELIABILITY COMPUTATION FOR PUF {idPuf}-----------------')
    # Compute the number of unique devices
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
    """
    Computes the bit aliasing metric for a given PUF (Physically Unclonable Function) 
    and generates corresponding plots and statistics.
    Bit aliasing measures the average response of each bit position across multiple 
    devices and repetitions for a given set of challenges. This function calculates 
    the bit aliasing for each challenge and bit position, generates a heatmap plot, 
    and saves the results to a file.
    Args:
        idTest (int): Identifier for the test being performed.
        df (numpy.ndarray): A 4D array containing the response data with dimensions 
            (numChals, respWidth, numReps, numDevice), where:
            - numChals: Number of challenges.
            - respWidth: Width of the response in bits.
            - numReps: Number of repetitions.
            - numDevice: Number of devices.
        idPuf (int): Identifier for the PUF being analyzed.
        indexDevToIdDev (dict): A mapping from device indices to device identifiers.
    Side Effects:
        - Creates directories for saving plots and statistics if they do not exist.
        - Generates and saves a heatmap plot of the bit aliasing metric.
        - Saves the computed mean and standard deviation of the bit aliasing metric 
          to a text file.
    Outputs:
        - Heatmap plot saved as a PDF file in the directory:
          `<baseTestsDir>/<idTest>/results/plots/bitaliasing/`
        - Statistics (mean and standard deviation) saved in a text file in the directory:
          `<baseTestsDir>/<idTest>/results/stats/bitaliasing/`
    Notes:
        - The x-axis of the heatmap is inverted so that the most significant bit (MSB) 
          is on the right and the least significant bit (LSB) is on the left.
        - The function prints progress and results to the console.
    Raises:
        - OSError: If there is an issue creating directories or saving files.
        - ValueError: If the input array `df` does not have the expected dimensions.
    """
    
    print(f'[CLIENT-APP] -----------------BIT ALIASING COMPUTATION FOR PUF {idPuf}-----------------')

    numDevice = df.shape[3]
    numReps = df.shape[2]
    respWidth = df.shape[1]
    numChals = df.shape[0]

    baseTestDir = baseTestsDir+str(idTest)
    # Creation of directories to save plots and statistics
    plotsPath = os.path.join(baseTestDir, 'results', 'plots' , 'bitaliasing')
    statsPath = os.path.join(baseTestDir, 'results', 'stats', 'bitaliasing')
    os.makedirs(plotsPath, exist_ok=True)
    os.makedirs(statsPath, exist_ok=True)

    bitAliasingFile = os.path.join(statsPath, f'bitAliasing_puf_{idPuf}.txt')
    # Remove the file if it exists
    if os.path.exists(bitAliasingFile):
        os.remove(bitAliasingFile)

    # allRespBitAlias will have dimensions (numChals, respWidth)
    allRespBitAlias = np.zeros((numChals, respWidth))

    # Compute the bit aliasing for each challenge
    for c in range(numChals):
        # Compute the average response bits for each challenge and device
        singleRespsBitAlias = np.nanmean(df[c, :, :, :], axis=(1, 2))
        allRespBitAlias[c, :] = singleRespsBitAlias     
        
    # Plotting the heatmap for bit aliasing
    plt.figure(figsize=(15, 2))

    hm = sns.heatmap(
        allRespBitAlias,
        cmap="viridis",
        cbar=True,
        xticklabels=False,  # Disable x tick labels
        yticklabels=True,  # (idem per y)
        vmin=0,
        vmax=1
    )

    # Customizing the heatmap
    cbar = hm.collections[0].colorbar
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    cbar.set_ticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])

    # Invert the x-axis so that the most significant bit (MSB) is on the right
    plt.gca().invert_xaxis()

    # Create tick positions for the x-axis
    numXTicks = 8
    tickPositions = np.linspace(0, respWidth - 1, numXTicks)

    # Compute the tick labels for the x-axis
    # The labels will be in reverse order so that:
    # - the column respWidth-1 (now on the left) has label "0" (LSB)
    # - the column 0 (now on the right) has label "respWidth-1" (MSB)
    tickLabels = [str(int(respWidth - 1 - x)) for x in tickPositions]
    plt.xticks(tickPositions, tickLabels, fontsize=16)

    # Set y-ticks to show challenge indices
    y_ticks = np.linspace(0, numChals-1, 1, dtype=int)
    plt.yticks(ticks=y_ticks, labels=[f"{i}" for i in y_ticks])
    plt.yticks(fontsize=14)
    plt.xlabel(f'Bit position', fontsize=18)
    plt.ylabel('Chal. index', fontsize=18)
    # plt.title(f'Bit Aliasing for PUF {idPuf}')

    # Save the heatmap plot
    plotFile = os.path.join(plotsPath, f'bit_aliasing_puf_{idPuf}.pdf')
    plt.savefig(plotFile, format='pdf', bbox_inches='tight', dpi=600)
    plt.close()

    # Compute the mean and standard deviation of the bit aliasing
    mu = np.nanmean(allRespBitAlias)
    std = np.nanstd(allRespBitAlias)
    print(f'[CLIENT-APP] - Total Bit Aliasing: mean = {mu} \t std = {std}')

    # Save the statistics to the file
    with open(bitAliasingFile, 'a') as f:
        f.write('Total\n')
        f.write(f'Mean: {mu}\n')
        f.write(f'Standard Deviation: {std}\n')

    print(f'[CLIENT-APP] - Bit aliasing plots saved in: {plotsPath}')

def ComputeMinHentropyDensity(idTest, df, idPuf, indexDevToIdDev):
    """
    Computes the minimum entropy density (H-min) for a given PUF (Physically Unclonable Function) 
    based on the provided response data. The function calculates the entropy for each challenge 
    and saves the results to a file.
    Args:
        idTest (int): Identifier for the test being performed.
        df (numpy.ndarray): A 4-dimensional array containing response data with dimensions 
            (numChals, respWidth, numReps, numDevice), where:
            - numChals: Number of challenges.
            - respWidth: Width of the response.
            - numReps: Number of repetitions.
            - numDevice: Number of devices.
        idPuf (int): Identifier for the PUF being analyzed.
        indexDevToIdDev (dict): A mapping from device indices to device identifiers.
    Side Effects:
        - Creates directories for saving plots and statistics if they do not already exist.
        - Deletes the existing HMinDensity file for the given PUF if it exists.
        - Writes the computed H-min value for the PUF to a file.
    Notes:
        - The function calculates the average response bits for each challenge and device, 
          determines the reference responses, and computes the minimum entropy density 
          for each challenge.
        - The entropy is calculated using the formula: Hmin = -log2(pb_max), where pb_max 
          is the maximum probability of a bit being 0 or 1.
    Outputs:
        - A text file named `HMinDensity_puf_<idPuf>.txt` containing the computed H-min value 
          for the given PUF, saved in the `stats/HMinDensity` directory under the test results.
    """
    
    print(f'[CLIENT-APP] -----------------H-min Density COMPUTATION FOR PUF {idPuf}-----------------')

    numDevice = df.shape[3]
    numReps = df.shape[2]
    respWidth = df.shape[1]
    numChals = df.shape[0]

    baseTestDir = baseTestsDir+str(idTest)
    # Creation of directories to save plots and statistics
    plotsPath = os.path.join(baseTestDir, 'results', 'plots' , 'HMinDensity')
    statsPath = os.path.join(baseTestDir, 'results', 'stats', 'HMinDensity')
    os.makedirs(plotsPath, exist_ok=True)
    os.makedirs(statsPath, exist_ok=True)

    HMinDensityFile = os.path.join(statsPath, f'HMinDensity_puf_{idPuf}.txt')
    # Remove the file if it exists
    if os.path.exists(HMinDensityFile):
        os.remove(HMinDensityFile)

    # allRespBitAlias will have dimensions (numChals, respWidth, numDevice)
    referenceResps = np.zeros((numChals, respWidth, numDevice))

    for c in range(numChals):
        for d in range(numDevice):
            averageRespBits = np.nanmean(df[c, :, :, d], axis=1)
            referenceResps[c, :, d] = np.where(averageRespBits > 0.5, 1, 0)
        
        # Compute the H-min for the current challenge
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
    """
    Computes the uniqueness metric for a given PUF (Physical Unclonable Function) 
    based on the provided response data. The uniqueness metric evaluates the 
    differences in responses between devices for the same challenges.
    Parameters:
    -----------
    idTest : int
        Identifier for the test being performed.
    df : numpy.ndarray
        A 4-dimensional array containing the response data. The dimensions are 
        expected to be (numChals, respWidth, numReps, numDevice), where:
        - numChals: Number of challenges.
        - respWidth: Width of the response.
        - numReps: Number of repetitions.
        - numDevice: Number of devices.
    idPuf : int
        Identifier for the PUF being analyzed.
    indexDevToIdDev : dict
        A mapping from device indices to device identifiers.
    Outputs:
    --------
    - Saves boxplot and histogram visualizations of uniqueness metrics to the 
      specified directory.
    - Writes the mean and standard deviation of the uniqueness metrics to a 
      text file.
    Notes:
    ------
    - The function calculates the Hamming distance between the responses of 
      different devices for each challenge.
    - NaN values in the response data are handled appropriately to ensure 
      valid computations.
    - The results are saved in a directory structure based on the test ID.
    Raises:
    -------
    - Ensures that the output directories exist and removes any pre-existing 
      uniqueness files before writing new results.
    Visualization:
    --------------
    - Boxplot: Displays the distribution of uniqueness values across challenges.
    - Histogram: Shows the density of uniqueness values across all challenges.
    Example Usage:
    --------------
    ComputeUniqueness(idTest=1, df=response_data, idPuf=42, indexDevToIdDev=device_mapping)
    """
    
    print(f'[CLIENT-APP] -----------------UNIQUENESS COMPUTATION FOR PUF {idPuf}-----------------')
    numDevice = df.shape[3]
    numReps = df.shape[2]
    respWidth = df.shape[1]
    numChals = df.shape[0]

    baseTestDir = baseTestsDir + str(idTest)
    # Creazione di directory per salvare i grafici e le statistiche
    plotsPath = os.path.join(baseTestDir, 'results', 'plots', 'uniqueness')
    statsPath = os.path.join(baseTestDir, 'results', 'stats', 'uniqueness')

    uniquenessFile = os.path.join(statsPath, f'uniqueness_puf_{idPuf}.txt')
    # Remove the file if it exists
    if os.path.exists(uniquenessFile):
        os.remove(uniquenessFile)

    os.makedirs(plotsPath, exist_ok=True)
    os.makedirs(statsPath, exist_ok=True)

    # List to collect all uniqueness values across challenges
    allUniqueness = []
    uniqueness_per_challenge = []

    for c in range(numChals):
        
        # Lis to collect uniqueness values for the current challenge
        chUniqueness = []

        print(f"df shape: {df.shape}")
        for d in range(numDevice):

            meanResponseA = np.nanmean(df[c, :, :, d], axis=1)
            referenceRespA = np.where(meanResponseA > 0.5, 1, 0)

            for j in range(d + 1, numDevice):

                meanResponseB = np.nanmean(df[c, :, :, j], axis=1)
                referenceRespB = np.where(meanResponseB > 0.5, 1, 0)

                diffResps = np.bitwise_xor(referenceRespA, referenceRespB)
                countDiff = np.count_nonzero(diffResps)
                chUniqueness.append(countDiff / respWidth)

        # Compute the uniqueness for the current challenge
        if chUniqueness:
           uniqueness_per_challenge.append(np.mean(chUniqueness))
        else:
           uniqueness_per_challenge.append(np.nan)  # Assegna NaN se non ci sono confronti validi        
                  
        allUniqueness.append(uniqueness_per_challenge)

    # Generate boxplot for uniqueness per challenge
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

    # Generate histogram for uniqueness distribution
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
    """
    Analyzes frequency data for a given test and device, generating heatmaps and statistics 
    for each PUF (Physically Unclonable Function) in the provided list.
    Parameters:
        idTest (int): Identifier for the test.
        idDev (int): Identifier for the device.
        idPufList (list): List of PUF IDs to process.
        df (pandas.DataFrame): DataFrame containing the frequency data. 
            Expected columns include 'counter1', 'counter2', 'idpuf', 'challenge', and 'temperature'.
        oscillPeriod (float): Oscillation period used to calculate frequencies.
    Returns:
        None
    Side Effects:
        - Creates directories to save plots and statistics.
        - Saves heatmaps as PDF files in the 'plots/frequency' directory.
        - Writes statistical summaries to a text file in the 'stats/frequency' directory.
        - Prints progress and summary statistics to the console.
    Notes:
        - The function computes frequencies from hexadecimal counters and calculates the 
          difference between two frequencies ('delta_freq').
        - Heatmaps are generated for each PUF, showing the relationship between challenges 
          and temperatures.
        - Statistical summaries include the mean and standard deviation of delta frequencies 
          for each PUF.
    Example:
        AnalyzeFrequencies(
            idTest=1,
            idDev=2,
            idPufList=[101, 102],
            df=dataframe,
            oscillPeriod=0.01
    """
    
    print(f'[CLIENT-APP] -----------------FREQUENCIES COMPUTATION FOR DEV {idDev}-----------------')
    
    # Create directories to save plots and statistics
    baseTestDir = baseTestsDir + str(idTest)
    plotsPath = os.path.join(baseTestDir, 'results', 'plots', 'frequency')
    statsPath = os.path.join(baseTestDir, 'results', 'stats', 'frequency')
    frequencyFile = os.path.join(statsPath, f'frequency.txt')

    os.makedirs(plotsPath, exist_ok=True)
    os.makedirs(statsPath, exist_ok=True)

    # Convert hexadecimal counters to integers and calculate frequencies
    df['counter1_int'] = df['counter1'].apply(lambda x: int(x, 16))
    df['counter2_int'] = df['counter2'].apply(lambda x: int(x, 16))
    df['freq1'] = df['counter1_int'] / oscillPeriod
    df['freq2'] = df['counter2_int'] / oscillPeriod
    df['delta_freq'] = df['freq1'] - df['freq2']

    for idPuf in idPufList:
        print(f'[CLIENT-APP] Processing PUF ID: {idPuf}')
        
        # Filter the DataFrame for the current PUF ID
        df_idPuf = df[df['idpuf'] == idPuf]
        
        # order the DataFrame by challenge and temperature
        challenges = sorted(df_idPuf['challenge'].unique())
        temperatures = sorted(df_idPuf['temperature'].unique())
        
        # Create a matrix to hold the heatmap data
        heatmap_data = np.full((len(challenges), len(temperatures)), np.nan)
        challenge_to_idx = {challenge: idx for idx, challenge in enumerate(challenges)}
        temp_to_idx = {temp: idx for idx, temp in enumerate(temperatures)}

        # Populate the heatmap data
        for _, row in df_idPuf.iterrows():
            challenge_idx = challenge_to_idx[row['challenge']]
            temp_idx = temp_to_idx[row['temperature']]
            heatmap_data[challenge_idx, temp_idx] = row['delta_freq']

        # Limit the number of challenges to 20 for better visualization
        # step = max(1, len(challenges) // 20)
        # yticklabels = [challenges[i] if i % step == 0 else "" for i in range(len(challenges))]
        step = max(1, len(challenges) // 10)
        yticks = [i for i in range(len(challenges)) if i % step == 0]
        yticklabels = [challenges[i] for i in yticks]

        # Create the heatmap
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

        # Modify the font size
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

        # Save the heatmap plot
        plotFile = os.path.join(plotsPath, f'frequency_dev_{idDev}_puf_{idPuf}.pdf')
        plt.savefig(plotFile, format='pdf', bbox_inches='tight', dpi=300)
        plt.close()

        # Compute statistics for the delta frequencies
        mean_per_challenge = []
        std_per_challenge = []
        for challenge in challenges:
            df_challenge = df_idPuf[df_idPuf['challenge'] == challenge]
            mean_per_challenge.append(df_challenge['delta_freq'].mean())
            std_per_challenge.append(df_challenge['delta_freq'].std())

        # Average the means and standard deviations across challenges
        mean_of_means = np.mean(mean_per_challenge)
        mean_of_stds = np.mean(std_per_challenge)

        # Print and save the statistics
        print(f'[CLIENT-APP] - PUF {idPuf}: Mean of Delta Freq = {mean_of_means:.4f}, Std of Delta Freq = {mean_of_stds:.4f}')
        with open(frequencyFile, 'a') as f:
            f.write(f'Device {idDev}\n')
            f.write(f'ID PUF: {idPuf}\n')
            f.write(f'Mean of Delta Frequencies: {mean_of_means:.4f}\n')
            f.write(f'Standard Deviation of Delta Frequencies: {mean_of_stds:.4f}\n\n')

    print(f'[CLIENT-APP] - Frequencies analysis completed. Plots saved in: {plotsPath}')
    """
    Computes the bit reliability and uniqueness of a Physical Unclonable Function (PUF) 
    across multiple devices and temperature variations. The function generates heatmaps, 
    reliability plots, and uniqueness plots for the given PUF data.
    Parameters:
    -----------
    idTest : int
        Identifier for the test being conducted.
    df : numpy.ndarray
        A 4D array containing PUF responses with dimensions 
        (numChals, respWidth, numReps, numDevice). Each element represents 
        the response of a device to a specific challenge under certain conditions.
    idPuf : int
        Identifier for the PUF being analyzed.
    indexDevToIdDev : dict
        A mapping from device indices to device identifiers.
    Returns:
    --------
    None
        The function saves the computed results (heatmaps, reliability plots, 
        and uniqueness plots) to the appropriate directories.
    Notes:
    ------
    - The function skips computation if the number of challenges exceeds 10.
    - Reliability is computed for each device and challenge, normalized, and visualized 
      as heatmaps.
    - Reliability and uniqueness are analyzed across temperature clusters, and the results 
      are plotted.
    - The function creates directories for saving plots and statistics if they do not exist.
    - Results are saved as PDF files in the specified directories.
    Outputs:
    --------
    - Heatmaps for bit reliability per challenge.
    - Line plots for reliability across temperature variations for each device.
    - Line plots for uniqueness across temperature clusters.
    - Combined plots for reliability and uniqueness against temperature clusters.
    Raises:
    -------
    - The function assumes the existence of global variables `baseTestsDir`, `deltaT_cluster`, 
      and `TemperatueArray`. Ensure these are defined before calling the function.
    - If the input data contains NaN values, they are handled appropriately during computations.
    """
    
    print(f'[CLIENT-APP] -----------------BIT RELIABILITY COMPUTATION FOR PUF {idPuf}-----------------')

    numDevice = df.shape[3]
    numReps = df.shape[2]
    respWidth = df.shape[1]
    numChals = df.shape[0]

    if numChals > 10:
        print(f'[CLIENT-APP] - Too many challenges ({numChals}). Skipping bit reliability computation.')
        return

    baseTestDir = baseTestsDir+str(idTest)
    # Creation of directories to save plots and statistics
    plotsPath = os.path.join(baseTestDir, 'results', 'plots' , 'reliability')
    statsPath = os.path.join(baseTestDir, 'results', 'stats', 'reliability')
    os.makedirs(plotsPath, exist_ok=True)
    os.makedirs(statsPath, exist_ok=True)

    bitReliabilityFile = os.path.join(statsPath, f'bitreliability_puf_{idPuf}.txt')
    # Remove the file if it exists
    if os.path.exists(bitReliabilityFile):
        os.remove(bitReliabilityFile)

    # Compute the bit reliability for each challenge
    for c in range(numChals):
        # allRespBitAlias will have dimensions (numDevice, respWidth)
        allDevAllChBitRel = np.zeros((numDevice, respWidth))
        # Compute the average response bits for each challenge and device
        for d in range(numDevice):
            # Average the responses across repetitions for each device and challenge
            oneDevOneChBitRel = np.nanmean(df[c, :, :, d], axis=(1))
            allDevAllChBitRel[d, : ] = oneDevOneChBitRel

        normalizedBitReliability = np.where(allDevAllChBitRel > 0.5, allDevAllChBitRel, 1 - allDevAllChBitRel)

        # Plotting the heatmap for bit reliability
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

        # Customizing the heatmap
        cbar = hm.collections[0].colorbar
        cbar.set_ticks([0.5, 0.6, 0.7, 0.8, 0.9, 1])
        cbar.set_ticklabels(['0.5', '0.6', '0.7', '0.8', '0.9', '1'])
   
        # Invert the x-axis so that the most significant bit (MSB) is on the right
        plt.gca().invert_xaxis()

        # Invert the y-axis so that the first device is at the top
        plt.gca().invert_yaxis()

        # Create tick positions for the x-axis
        numXTicks = 6
        tickPositions = np.linspace(0, respWidth - 1, numXTicks)

        # Compute the tick labels for the x-axis
        # The labels will be in reverse order so that:
        # - the column respWidth-1 (now on the left) has label "0" (LSB)
        # - the column 0 (now on the right) has label "respWidth-1" (MSB)
        tickLabels = [str(int(respWidth - 1 - x)) for x in tickPositions]
        plt.xticks(tickPositions, tickLabels)

        plt.xlabel('Bit', fontsize=14)
        plt.ylabel('Device Index', fontsize=14)

        # Save the heatmap plot
        plotFile = os.path.join(plotsPath, f'bit_reliability_{idPuf}_ch_{c}.pdf')
        plt.savefig(plotFile, format='pdf', bbox_inches='tight', dpi=300)
        plt.close()

    # Calcoliamo mean e std sull'intera matrice
    mu = np.nanmean(allDevBitRel)
    std = np.nanstd(allDevBitRel)
    print(f'[CLIENT-APP] - Total Bit Aliasing: mean = {mu} \t std = {std}')

    # Salvataggio dei risultati su file
    with open(bitReliabilityFile, 'a') as f:
        f.write('Total\n')
        f.write(f'Mean: {mu}\n')
        f.write(f'Standard Deviation: {std}\n')

    print(f'[CLIENT-APP] - BIT Reliability plots saved in: {plotsPath}')