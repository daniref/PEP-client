import argparse

import core.Utility as UT
import core.Handlers.HandlerTests as HT
import core.Handlers.HandlerResults as HR
import core.Handlers.HandlerPower as HP
import core.Handlers.HandlerAccess as HA

if __name__ == '__main__':
    UT.DisplayLogo()

    parser = argparse.ArgumentParser(description="Automated client script to manage FPGA-based PUF experiments.",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("op", type=int,
                        help="""0: Launch experiments
1: Download experiment results
2: Calculate quality metrics
3: Power up devices
4: Power down devices
5: Power up fans
6: Power down fans
7: Register new user""")
    parser.add_argument("--numDevices", type=int,
                        help="Number of devices to use (required for op=0)")
    parser.add_argument("--idTest", type=int,
                        help="ID of the test (required for op in [0,1,2])")
    parser.add_argument("--serverAddress", type=str, choices={"pub", "priv"}, required=True,
                        help="Use 'pub' for Internet, 'priv' for local SPECTRE-net")
    parser.add_argument("--username", type=str, required=True,
                        help="Your username")
    parser.add_argument("--password", type=str, required=True,
                        help="Your password")

    args = parser.parse_args()

    if not UT.SetServerAddress(args.serverAddress):
        parser.error("Invalid server address")

    result, idUsr = HA.LogInUser(args.username, args.password)
    if not result:
        print("[CLIENT-AUTO] - Login failed!")
        exit(1)

    if args.op == 0:
        if args.numDevices is None or args.idTest is None:
            parser.error("Both --numDevices and --idTest are required for op=0")
        HT.LaunchTests(args.numDevices, args.idTest, idUsr, args.username, args.password)

    elif args.op == 1:
        if args.idTest is None:
            parser.error("--idTest is required for op=1")
        HR.ExpCampaignDownloading(args.idTest, args.username, args.password)

    elif args.op == 2:
        if args.idTest is None:
            parser.error("--idTest is required for op=2")
        if HR.CheckCRPsExist(args.idTest):
            print("[CLIENT-AUTO] - CRPs already downloaded.")
        else:
            print("[CLIENT-AUTO] - CRPs missing, starting download.")
            HR.ExpCampaignDownloading(args.idTest, args.username, args.password)
        HR.CalculateMetrics(args.idTest)

    elif args.op == 3:
        if HP.PowerUp('device', args.username, args.password):
            print("[CLIENT-AUTO] - Devices powered up.")
        else:
            print("[CLIENT-AUTO] - Error powering up devices.")

    elif args.op == 4:
        if HP.PowerDown('device', args.username, args.password):
            print("[CLIENT-AUTO] - Devices powered down.")
        else:
            print("[CLIENT-AUTO] - Error powering down devices.")

    elif args.op == 5:
        if HP.PowerUp('fan', args.username, args.password):
            print("[CLIENT-AUTO] - Fans powered up.")
        else:
            print("[CLIENT-AUTO] - Error powering up fans.")

    elif args.op == 6:
        if HP.PowerDown('fan', args.username, args.password):
            print("[CLIENT-AUTO] - Fans powered down.")
        else:
            print("[CLIENT-AUTO] - Error powering down fans.")

    elif args.op == 7:
        HA.RegisterUser()

    else:
        parser.error("Invalid operation: op must be in range [0-7]")
