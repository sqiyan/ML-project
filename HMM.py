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
            self.path = "SG"
        else:
            self.path = "EN"
        self.open_file()

    def open_file(self):
        g = open(self.path+"/train", encoding="utf-8")
        test_data_list = g.read().splitlines()
        test_data_list_formatted = []
        for line in test_data_list:
            entries = line.split()
            test_data_list_formatted.append(np.array(entries, dtype=str))
        self.train_data = np.array(test_data_list_formatted)
        g.close()
        
        f = open(self.path+"/dev.in", encoding="utf-8")
        test_data_list = f.read().splitlines()
        self.test_data = np.array(test_data_list)
        f.close()

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
        self.ymax_given_x()
        f = open(self.path + "/dev.p2.out","w", encoding="utf-8")
        for x in self.test_data:
            if len(x)<1:
                f.write("\n")
            else:
                try:
                    y = self.y_max_given_x[x]
                    f.write("{} {}\n".format(x,y))
                except:
                    y = self.y_max_given_x["#UNK#"]
                    f.write("{} {}\n".format(x,y))
        f.close()
        # print(len(self.test_data))




hmm = HMM_script(args)
hmm.evaluate_ymax()
# print(hmm.test_data)
# print(hmm.train_data)