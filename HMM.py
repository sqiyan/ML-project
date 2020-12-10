import numpy as np
import argparse as ap
import json
import pickle
from part2 import part2
# from part3 import part3

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
        """Returns in the form of a Dictionary, where {(x_val,y_val):probability},"""
        emission_obj = part2(self.test_data, self.train_data, self.path)
        emission_params = emission_obj.get_emission_params()
        self.picklize(emission_params, "em_params")
        if self.action == "eval" and self.part == 2:
            emission_obj.evaluate_ymax()
        return emission_params

    def part3_transition_params(self):
        """Returns in the form of a Dictionary, where {(prev_y,y):probability},"""
        # to be filled in
        
    # to convert to pickle format
    def picklize(self, object, name):
        """Writes a pickle with name: 'name + path'"""
        pickle.dump(object, open(name+self.path + ".p","wb"))

    def load_pickle(self, name):
        """Loads pickle with name: 'name + path'. Returns object."""
        return pickle.load(open(name+self.path + ".p","rb"))

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

    # make test data iterable for viterbi
    def format_testdata(self):
        test_sequences = [[]]
        test_data = self.test_data
        i = 0
        for line in test_data:
            if line == "": # indicates a new sequence
                test_sequences.append([])
                i += 1
            else:
                test_sequences[i].append(line.strip('\n'))
        test_sequences.pop() # remove space in last line
        self.input_sequences = test_sequences

    def mini_viterbi(self, input_sequence):

        states = self.states
        transition_dict = self.p_y1_given_y0

        # Opening JSON file 
        f = open('em_params_EN.json',) 
        
        # returns JSON object as  
        # a dictionary 
        emission_dict = json.load(f)

        emission_dict = self.e_x_given_y

        # take unique states in input data
        all_states = states[1:len(states)]
        n = len(input_sequence)

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
                    if transition not in transition_dict.keys():
                        p = 0
                    else:
                        p = sequence_prob[layer - 1][previous_state]["p"] * \
                        transition_dict[(transition)]
                    if p > max_p:
                        max_p = p
                        print("updating prev state")
                        max_prob_prev_state = previous_state
                sequence_prob[layer]["STOP"] = {"p": max_p, "previous": max_prob_prev_state}
                continue

            for current_state in sequence_prob[layer]:
                max_p = 0
                max_prob_prev_state = "NA"
                for previous_state in sequence_prob[layer - 1]:
                    transition = (previous_state, current_state)


                    if (input_sequence[layer - 1], current_state) in emission_dict.keys() and transition in transition_dict.keys():
                        p = sequence_prob[layer - 1][previous_state]["p"] * \
                            transition_dict[(transition)] * \
                            emission_dict[(input_sequence[layer - 1], current_state)]
                    elif transition in transition_dict.keys():
                        p = sequence_prob[layer - 1][previous_state]["p"] * \
                            transition_dict[(transition)] * \
                            0.00000000000001 # to allow initial state "NA" to be updated
                    else:
                        p = sequence_prob[layer - 1][previous_state]["p"] * \
                            0.00000000000001 * \
                            0.00000000000001 # to allow initial state "NA" to be updated
                    if p > max_p:
                        max_p = p
                        max_prob_prev_state = previous_state
                sequence_prob[layer][current_state] = {"p": max_p, "previous": max_prob_prev_state}

        # backtracking to find argmax
        current_layer = n
        reverse_path = ["STOP"]
        while current_layer >= 0:
            reverse_path.append(sequence_prob[current_layer + 1][reverse_path[len(reverse_path) - 1]]["previous"])
            # just means taking the current state being backtracked, find its most probable previous state as argmax
            current_layer -= 1

        return reverse_path[::-1][1:len(reverse_path)-1]

    def viterbi(self):
        self.format_testdata()
        input_sequences = self.input_sequences
        # print(input_sequences)
        pred_state_sequence = [[]]
        i = 0
        for input_sequence in input_sequences:
            for state in self.mini_viterbi(input_sequence):
                pred_state_sequence[i].append(state)
            pred_state_sequence.append([]) # to store state sequence for next sentence
            i += 1
        pred_state_sequence.pop() # remove last []
        return pred_state_sequence


hmm = HMM_script(args)
# hmm.evaluate_ymax()
print(hmm.est_transition_params())
# hmm.save_tr_to_json()
print("hello")
# print(hmm.e_x_given_y)
print(hmm.viterbi())
# print(hmm.test_data)
# hmm.evaluate_ymax()
# hmm.save_to_json()
# print(hmm.est_transition_params())
# print(hmm.test_data)
# print(hmm.train_data)
# hmm.est_transition_params()

# for line in hmm.train_data:
#     if len(line) == 0:
#         print("hello")


hmm = HMM_script(args)
if int(args.part) ==2:
    print(hmm.part2_emission_params())
elif int(args.part) ==3:
    # print(hmm.part3_transition_params())
    hmm.part3_transition_params()
else:
    print("add in parts 3 and 4 here")
print("Part {} complete. {}-ed on {} test set.".format(args.part,args.action,hmm.path))
