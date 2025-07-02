import core.Utility as UT
import core.AccessData as AD
import core.Handlers.HandlerTests as HT
import core.Handlers.HandlerResults as HR
import core.Handlers.HandlerPower as HP
import core.Handlers.HandlerAccess as HA


if __name__ == '__main__':
    UT.DisplayLogo()

    serverAddress = AD.access
    if(True == UT.SetServerAddress(serverAddress)):

        username = AD.userName
        password = AD.password

        result,idUsr=HA.LogInUser(username,password)
        if(result != True):
            print("Log in request denied!")
        else:

            # Ask the user to input the initial value
            value = int(input("""Choose an operation:
            0 to launch experiments on devices;
            1 to retrieve experiment results;
            2 to calculate quality metrics;
            3 to power on the devices;
            4 to power off the devices;
            5 to power on the fans;
            6 to power off the fans:
            
        Operation to perform: """))

            # Execute the function based on the chosen value
            if value == 0:
                numDevices = int(input("Enter the number of devices to use for experiments: "))
                idTest = int(input("Enter the Test ID: "))
                HT.LaunchTests(numDevices, idTest,idUsr,username,password)

            elif value == 1:
                idTest = int(input("Enter the Test ID: "))
                HR.ExpCampaignDownloading(idTest,username,password)
            
            elif value == 2:
                idTest = int(input("Enter the Test ID: "))

                if(HR.CheckCRPsExist(idTest)):
                    print("[CLIENT-APP] - CRPs already downloaded!")
                else:
                    print("[CLIENT-APP] - CRPs not present, downloading started!")
                    HR.ExpCampaignDownloading(idTest,username,password)

                HR.CalculateMetrics(idTest)
            
            elif value == 3:
                if HP.PowerUp('device',username,password):
                    print("[CLIENT-APP] - All devices have been powered on!")
                else:
                    print("[CLIENT-APP] - Error powering on the devices!")
            
            elif value == 4:
                if HP.PowerDown('device',username,password):
                    print("[CLIENT-APP] - All devices have been powered off!")
                else:
                    print("[CLIENT-APP] - Error powering off the devices!")

            elif value == 5:
                if HP.PowerUp('fan',username,password):
                    print("[CLIENT-APP] - All fans have been powered on!")
                else:
                    print("[CLIENT-APP] - Error powering on the fans!")
            
            elif value == 6:
                if HP.PowerDown('fan',username,password):
                    print("[CLIENT-APP] - All fans have been powered off!")
                else:
                    print("[CLIENT-APP] - Error powering off the fans!")

            else:
                print("Accepted values are within the range [0:6]")
    else:
        print("Server choice not accepted!")
