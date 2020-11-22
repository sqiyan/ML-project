import numpy as np
import argparse as ap

parser = ap.ArgumentParser(description='To run HMM on stuff')
parser.add_argument('--file', metavar='E', type=str, default='E',
                   help='Which file to run on. C for chinese, E for english and S for SG')


args = vars(parser.parse_args())


class HMM_script():
    def __init__(self, file):
        args = file['file']
        if args == "E":
            self.path = "EN"
        elif args == "C":
            self.path = "CN"
        elif args == "S":
            self.path = "SN"
        else:
            self.path = "EN"
        print(self.path)

    def open_file(self):
        self.data = np.genfromtxt(self.path + "/train",delimiter=" ",dtype = str)
        print(self.data)
        print(self.path + "/train")


hmm = HMM_script(args)
hmm.open_file()