import numpy as np
import argparse as ap
import json
import pickle
from part2 import part2
import part3

parser = ap.ArgumentParser(description='To run HMM on stuff')
parser.add_argument('--file', default='E',
                   help='Which file to run on. C for chinese, E for english and S for SG')
parser.add_argument('--part', default='2',
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
        emission_obj = part2(self.test_data, self.train_data, self.path)
        emission_params = emission_obj.get_emission_params()
        self.picklize(emission_params, "em_params")
        if self.action == "eval" and self.part == 2:
            emission_obj.evaluate_ymax()
        return emission_params
        
    # to convert to pickle format
    def picklize(self, object, name):
        """Writes a pickle with name: 'name + path'"""
        pickle.dump(object, open(name+self.path + ".p","wb"))

    def load_pickle(self, name):
        """Loads pickle with name: 'name + path'. Returns object."""
        return pickle.load(open(name+self.path + ".p","rb"))

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

    # part 3
    # estimate transition parameters from the training set using MLE 

    # count(u,v) Number of times we see a transition from u to v
    # count(u) Number of times we see the state u in the training set

    def est_transition_params(self):
        self.est_emission_params()
        count_y = {}
        count_y0_to_y1 = {}
        previous_state = None
        self.p_y1_given_y0 = {}
        # result = str.split('\n\n')
        # result = result[:-1]
        
        # for entry in result: 
        for line in self.train_data:
            if previous_state == None: 
                previous_state = "START"
                state = line[1]
                transition = (previous_state, state)
                count_y0_to_y1[(transition)] = count_y0_to_y1[(transition)] + 1 if transition in count_y0_to_y1 else 1
                count_y["START"] = count_y["START"] + 1 if "START" in count_y else 1

            elif line[1] == "":
                state = "STOP"
                transition = (previous_state, state)
                count_y0_to_y1[(transition)] = count_y0_to_y1[(transition)] + 1 if transition in count_y0_to_y1 else 1
                previous_state = None

            else:
                state = line[1]
                transition = (previous_state, state)
                count_y0_to_y1[(transition)] = count_y0_to_y1[(transition)] + 1 if transition in count_y0_to_y1 else 1
                count_y[state] = count_y[state] + 1 if state in count_y else 1
                previous_state = state

        # y_previous_state = None
        # for entry in y_vals:
        #     if previous_state == None:
        #         entry = "START"
        #         if entry not in(count_y):
        #             count_y[entry] = 1
        #         else:
        #             count_y[entry] +=1
        #         previous_state = entry
        #     elif entry == "":
        #         entry = "STOP"
        #         if entry not in(count_y):
        #             count_y[entry] = 1
        #         else:
        #             count_y[entry] +=1
        #         previous_state = entry
        #     else:
        #         if entry not in(count_y):
        #             count_y[entry] = 1
        #         else:
        #             count_y[entry] +=1
        #         previous_state = entry

        for entry, count in count_y0_to_y1.items():
            try:
                self.p_y1_given_y0[entry] = count/count_y[entry[0]]
            except:
                self.p_y1_given_y0[entry] = count/count_y[entry[0]]

        self.argmax_tr = self.p_y1_given_y0

        return self.argmax_tr

    # Use the estimated transition and emission parameters, implement the Viterbi algorithm
    # Report the precision, recall and F scores of all systems

    # TODO - implement dictionary to store max and argmax values

    def format_testdata():
        test_data = self.test_data
        

    def viterbi(self):
        
        observed_sequence = self.test_data

        states = self.states
        print(states)
        transition_dict = self.p_y1_given_y0

        # Opening JSON file 
        f = open('em_params_EN.json',) 
        
        # returns JSON object as  
        # a dictionary 
        emission_dict = json.load(f)

        # take from unique states in input data
        all_states = states[1:len(states)]
        n = len(observed_sequence)

        # instantiate a nested dictionary to store and update values of sequence probability
        sequence_prob = {0: {"START": {"p": 1.0, "previous": "NA"}}}

    
    # TODO - write code to update most probable transmission and emission sequence for given sentence

        for i in range(n):
            sequence_prob[i + 1] = {}
        for layer in sequence_prob:
            if layer == 0: continue
            for state in all_states:
                sequence_prob[layer][state] = {}
        sequence_prob[n + 1] = {"STOP": {}}

        for layer in sequence_prob:
            if layer == 0: continue

            if layer == n + 1:
                max_p = 0
                max_prob_prev_state = "NA"
                for previous_state in sequence_prob[layer - 1]:
                    transition = (previous_state, "STOP")
                    p = sequence_prob[layer - 1][previous_state]["p"] * \
                        transition_dict[(transition)]
                    if p > max_p:
                        max_p = p
                        max_prob_prev_state = previous_state
                sequence_prob[layer]["STOP"] = {"p": max_p, "previous": max_prob_prev_state}
                continue

            for current_state in sequence_prob[layer]:
                max_p = 0
                max_prob_prev_state = "NA"
                for previous_state in sequence_prob[layer - 1]:
                    transition = (previous_state, current_state)


                    if current_state in emission_dict[observed_sequence[layer - 1]]:
                        p = sequence_prob[layer - 1][previous_state]["p"] * \
                            transition_dict[(transition)] * \
                            emission_dict[observed_sequence[layer - 1]][current_state]
                    else:
                        p = sequence_prob[layer - 1][previous_state]["p"] * \
                            transition_dict[(transition)] * \
                            0.0000001
                    if p > max_p:
                        max_p = p
                        max_prob_prev_state = previous_state
                sequence_prob[layer][current_state] = {"p": max_p, "previous": max_prob_prev_state}

        # backtracking to find argmax
        current_layer = n
        reverse_path = ["STOP"]
        while current_layer >= 0:
            reverse_path.append(sequence_prob[current_layer + 1][reverse_path[len(reverse_path) - 1]]['previous'])
            current_layer -= 1

        return reverse_path[::-1][1:len(reverse_path)-1]



hmm = HMM_script(args)
if int(args.part) ==2:
    hmm.part2_emission_params()
else:
    print("add in parts 3 and 4 here")
print("Part {} complete. {}-ed on {} test set.".format(args.part,args.action,hmm.path))
