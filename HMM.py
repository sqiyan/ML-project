import numpy as np
import argparse as ap
import json
import pickle
from part2 import part2
from part3 import part3

parser = ap.ArgumentParser(description='To run HMM on stuff')
parser.add_argument('--file', default='E',
                   help='Which file to run on. C for chinese, E for english and S for SG')
parser.add_argument('--part', default='3', # i changed this to 3, but was initially 2
                   help='Which part to do. 2, 3, 4, 5')
parser.add_argument('--action', default='train',
                   help='train or eval')


args = parser.parse_args()


class HMM_script():

    # this is for SG and CN data
    non_EN_states = ["START", "B-negative", "B-neutral", "B-positive", "O", "I-negative", "I-neutral", "I-positive", "STOP"]
    EN_states =  ['START', 'B-NP', 'I-NP', 'B-VP', 'B-ADVP', 'B-ADJP', 'I-ADJP', 'B-PP', 'O', 'B-SBAR', 'I-VP', 'I-ADVP', 'B-PRT', 'I-PP', 'B-CONJP', 'I-CONJP', 'B-INTJ', 'I-INTJ', 'I-SBAR', 'B-UCP', 'I-UCP', 'B-LST', 'STOP']
    states = EN_states

    def __init__(self, args):
        language = args.file
        if language == "E":
            self.path = "EN"
        elif language == "C":
            self.path = "CN"
            self.states = self.non_EN_states
        elif language == "S":
            self.path = "SG"
            self.states = self.non_EN_states
        else:
            self.path = "EN"
        self.part = int(args.part)
        self.action = args.action
        self.open_file()

    def open_file(self):
        """generate data from files (train and test)"""
        g = open(self.path+"/train", encoding="utf-8")
        test_data_list = g.read().splitlines()
        test_data_list_formatted = []
        for line in test_data_list:
            entries = line.split()
            if len(entries)==1:
                entries.append("")
            elif len(entries)==0:
                entries.append("")
                entries.append("")
            elif len(entries)!=2:
                entries.pop(0)
            test_data_list_formatted.append(entries)
            # test_data_list_formatted.append(np.array(entries, dtype=str))
        # self.train_data = np.array(test_data_list_formatted)
        self.train_data = np.array(test_data_list_formatted)
        g.close()
        

        f = open(self.path+"/dev.in", encoding="utf-8")
        test_data_list = f.read().splitlines()
        self.test_data = np.array(test_data_list)
        f.close()

    def part2_emission_params(self):
        """Returns in the form of a Dictionary, where {(x_val,y_val):probability},"""
        emission_obj = part2(self.test_data, self.train_data, self.path, self.action)
        emission_params = emission_obj.get_emission_params()
        self.picklize(emission_params, "em_params")
        if self.action == "eval" and self.part == 2:
            emission_obj.evaluate_ymax()
        return emission_params

    def part3_transition_params(self):
        """Returns in the form of a Dictionary, where {(prev_y,y):probability},"""
        self.transition_obj = part3(self.states, self.test_data, self.train_data, self.path, self.action)
        transition_params = self.transition_obj.get_transition_params()
        self.picklize(transition_params, "tr_params")
        return transition_params
        
    # to convert to pickle format
    def picklize(self, object, name):
        """Writes a pickle with name: 'name + path'"""
        pickle.dump(object, open(name+self.path + ".p","wb"))

    def load_pickle(self, name):
        """Loads pickle with name: 'name + path'. Returns object."""
        return pickle.load(open(name+self.path + ".p","rb"))

    def part3_viterbi(self):
        transition_dict = self.part3_transition_params()
        emission_dict = self.part2_emission_params()
        self.transition_obj.set_em_params(emission_dict)
        predicted_sequences = self.transition_obj.viterbi()
        self.picklize(predicted_sequences,"viterbi")
        if self.action =="eval":
            self.transition_obj.write_sequences()
        return predicted_sequences

# print("hello")

hmm = HMM_script(args)
if int(args.part) ==2:
    print(hmm.part2_emission_params())
elif int(args.part) ==3:
    hmm.part3_transition_params()
    hmm.part3_viterbi()
else:
    print("add in parts 3 and 4 here")
print("Part {} complete. {}-ed on {} test set.".format(args.part,args.action,hmm.path))
