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
        self.open_file()

    def open_file(self):
        self.train_data = np.genfromtxt(self.path + "/train",delimiter=" ",dtype = str, encoding="utf-8", invalid_raise = False)
        self.test_data = np.genfromtxt(self.path + "/dev.in",delimiter=" ",dtype = str, encoding="utf-8", invalid_raise = False)

    def est_emission_params(self):
        y_vals = self.train_data[:,1]
        count_y = {}
        count_y_to_x = {}
        self.e_x_given_y = {}
        for i in range(self.train_data.shape[0]):
            transition = self.train_data[i,:]
            transition = tuple(transition)
            if transition not in count_y_to_x:
                count_y_to_x[transition] = 1
            else:
                count_y_to_x[transition] += 1
        
        for entry in y_vals:
            if entry not in(count_y):
                count_y[entry] = 1
            else:
                count_y[entry] +=1

        for entry, count in count_y_to_x.items():
            self.e_x_given_y[entry] = count/count_y[entry[1]]

        for entry, count in count_y.items():
            self.e_x_given_y[("#UNK#",entry)] = 0.5/(count+0.5)


    def ymax_given_x(self):
        self.est_emission_params()
        x_max_prob = {}
        self.y_max_given_x = {}  
        for entry, prob in self.e_x_given_y.items():
            if entry[0] not in x_max_prob:
                x_max_prob[entry[0]] = prob
                self.y_max_given_x[entry[0]] = entry[1]
            else:
                if x_max_prob[entry[0]]<prob:
                    x_max_prob[entry[0]] = prob
                    self.y_max_given_x[entry[0]] = entry[1]
        

    def evaluate_ymax(self):
        self.y_max_given_x()
        # f = open(self.path + "/dev.p2.out","w")
        #     f.write("{} {}\n".format(x,y))
        # f.close()




hmm = HMM_script(args)
hmm.ymax_given_x()