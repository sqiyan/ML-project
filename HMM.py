import numpy as np
import argparse as ap
import json

parser = ap.ArgumentParser(description='To run HMM on stuff')
parser.add_argument('--file', metavar='E', type=str, default='E',
                   help='Which file to run on. C for chinese, E for english and S for SG')


args = vars(parser.parse_args())


class HMM_script():

    # this is for SG and CN data
    non_EN_states = ["START", "B-negative", "B-neutral", "B-positive", "O", "I-negative", "I-neutral", "I-positive", "STOP"]
    EN_states =  ['START', 'B-NP', 'I-NP', 'B-VP', 'B-ADVP', 'B-ADJP', 'I-ADJP', 'B-PP', 'O', 'B-SBAR', 'I-VP', 'I-ADVP', 'B-PRT', 'I-PP', 'B-CONJP', 'I-CONJP', 'B-INTJ', 'I-INTJ', 'I-SBAR', 'B-UCP', 'I-UCP', 'B-LST', 'STOP']
    states = EN_states

    def __init__(self, file):
        args = file['file']
        if args == "E":
            self.path = "EN"
        elif args == "C":
            self.path = "CN"
            self.states = self.non_EN_states
        elif args == "S":
            self.path = "SG"
            self.states = self.non_EN_states
        else:
            self.path = "EN"
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

    def est_emission_params(self):
        """generates emission parameters based on training data, saves it as self.e_x_given_y"""
        try:
            self.y_vals = self.train_data[:,1]
        except:
            print(self.test_data.shape)
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
        
        for entry in self.y_vals:
            if entry not in(count_y):
                count_y[entry] = 1
            else:
                count_y[entry] +=1

        for entry, count in count_y_to_x.items():
            self.e_x_given_y[entry] = count/count_y[entry[1]]

        for entry, count in count_y.items():
            self.e_x_given_y[("#UNK#",entry)] = 0.5/(count+0.5)


    def ymax_given_x(self):
        """Generates the most likley y value given a particular x val, saves in a dictionary self.y_max_given_x"""
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
        
    # to convert to pickle format
    def save_em_to_json(self):
        with open('em_params_' + self.path + '.json', 'w', encoding='utf-8') as f:
            json.dump(self.y_max_given_x, f, ensure_ascii=False, indent=4)
        
    def save_tr_to_json(self):
        with open('tr_params_' + self.path + '.json', 'w', encoding='utf-8') as f:
            json.dump(self.argmax_tr, f, ensure_ascii=False, indent=4)
        

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
        print(states)
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
                            0
                    else:
                        p = sequence_prob[layer - 1][previous_state]["p"] * \
                            0 * \
                            0
                    if p > max_p:
                        max_p = p
                        max_prob_prev_state = previous_state
                sequence_prob[layer][current_state] = {"p": max_p, "previous": max_prob_prev_state}

        # backtracking to find argmax
        current_layer = n
        reverse_path = ["STOP"]
        while current_layer >= 0:
            print(sequence_prob)
            if reverse_path[len(reverse_path) - 1] == "NA":
                continue
            reverse_path.append(sequence_prob[current_layer + 1][reverse_path[len(reverse_path) - 1]]["previous"])
            # just means taking the current state being backtracked
            # 
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
print(hmm.test_data)
# hmm.evaluate_ymax()
# hmm.save_to_json()
# print(hmm.est_transition_params())
# print(hmm.test_data)
# print(hmm.train_data)
# hmm.est_transition_params()

# for line in hmm.train_data:
#     if len(line) == 0:
#         print("hello")