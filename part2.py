import pickle

from numpy.lib.npyio import load

class part2:
    """Class to evaluate part 2. 
    Constructor args:
        1. test_data
        2. training_data
        3. path 
        
    Methods:
        1. get_emission_params() - returns dictionary
        2. evaluate_ymax() - writes to file dev.p2.out"""

    def __init__(self, test_data, train_data, path):
        self.test_data = test_data
        self.train_data = train_data
        self.path = path

    def __est_emission_params(self):
        """generates emission parameters based on training data, saves it as self.e_x_given_y"""
        try:
            self.y_vals = self.train_data[:,1]
        except:
            print(self.test_data.shape)
        count_y = {}
        count_y_to_x = {}
        self.e_x_given_y = {}
        for i in range(self.train_data.shape[0]):
            emission = self.train_data[i,:]
            emission = tuple(emission)
            if emission not in count_y_to_x:
                count_y_to_x[emission] = 1
            else:
                count_y_to_x[emission] += 1
        
        for entry in self.y_vals:
            if entry not in(count_y):
                count_y[entry] = 1
            else:
                count_y[entry] +=1

        for entry, count in count_y_to_x.items():
            self.e_x_given_y[entry] = count/count_y[entry[1]]

        for entry, count in count_y.items():
            self.e_x_given_y[("#UNK#",entry)] = 0.5/(count+0.5)

    def __find_y_max_given_x(self):
        """Generates the most likley y value given a particular x val, saves in a dictionary self.y_max_given_x"""
        # self.est_emission_params()
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
        return self.y_max_given_x

    def get_emission_params(self):
        self.__est_emission_params()
        return self.e_x_given_y

    def get_emission_max_params(self):
        self.__est_emission_params()
        self.__find_y_max_given_x()
        return self.y_max_given_x

    def evaluate_ymax(self):
        """Writes the generated pairs for the dataset to dev.p2.out"""
        try:
            self.e_x_given_y = self.load_pickle("em_params")
            self.__find_y_max_given_x()
        except:
            self.__find_y_max_given_x()
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

    def load_pickle(self, name):
        """Loads pickle with name: 'name + path'. Returns object."""
        return pickle.load(open(name+self.path + ".p","rb"))