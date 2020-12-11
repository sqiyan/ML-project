import pickle
import random

class part4:

    """Class to evaluate part 3. 
    Constructor args:
        1. test_data
        2. training_data
        3. path 
        
    Methods:
        1. get_transition_params() - returns dictionary
        2. format_testdata() - returns input test_data as a nested list of words/sentences, each nested list is 1 sentence in the input data
        3. mini_viterbi(input_sequence, emission_dict) - returns predicted state sequence for 1 sentence
        4. viterbi() - returns nested list of predicted state sequence for all sentences within dev-in """

    def __init__(self, states, test_data, train_data, path, action , top_k =3):
        self.states = states
        self.test_data = test_data
        self.train_data = train_data
        self.path = path
        self.action = action
        self.k = top_k


    # part 4
    # estimate transition parameters from the training set using MLE 

    # count(u,v) Number of times we see a transition from u to v
    # count(u) Number of times we see the state u in the training set

    def __est_transition_params(self):
        """generates transition parameters based on training data, saves it as self.p_y1_given_y0"""
        count_y = {}
        count_y0_to_y1 = {}
        previous_state = None
        self.p_y1_given_y0 = {}
        
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

        for entry, count in count_y0_to_y1.items():
            try:
                self.p_y1_given_y0[entry] = count/count_y[entry[0]]
            except:
                self.p_y1_given_y0[entry] = count/count_y[entry[0]]

        return self.p_y1_given_y0

    def get_transition_params(self):
        self.__est_transition_params()
        return self.p_y1_given_y0

    # Part 4 
    # Use the estimated transition and emission parameters, implement the Viterbi algorithm
    # Report the precision, recall and F scores of all systems

    def __format_testdata(self):
        """makes test data iterable for viterbi
        returns input test_data as a nested list of words/sentences, 
        each nested list is 1 sentence in the input data
        """
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

    def __mini_viterbi(self, input_sequence, emission_dict, transition_dict):
        """
        runs viterbi algorithm from 1 sequence/sentence, 
        returns predicted state sequence for 1 sentence
        """

        states = self.states

        #implement dictionary to store max and argmax values

        # take unique states in input data
        all_states = states[1:len(states)]
        n = len(input_sequence)

        # instantiate a nested dictionary to store and update values of sequence probability
        sequence_prob = {0: {"START": queue(self.k, start=True)}}

    
    # update most probable transmission and emission sequence for given sentence

        for i in range(n):
            sequence_prob[i + 1] = {}
        for layer in sequence_prob:
            if layer == 0: continue
            for state in all_states:
                sequence_prob[layer][state] = queue(self.k)
        sequence_prob[n + 1] = {"STOP": queue(self.k)}

        for layer in sequence_prob:
            #START layer
            if layer == 0: continue

            #STOP layer
            if layer == n + 1:
                for previous_state in sequence_prob[layer - 1]:
                    transition = (previous_state, "STOP")
                    if transition not in transition_dict.keys():
                        p = 0
                    else:
                        p = sequence_prob[layer - 1][previous_state].get_max_prob() * \
                        transition_dict[(transition)]
                    sequence_prob[layer]["STOP"].try_add(p,previous_state)
                continue

            for current_state in sequence_prob[layer]:
                max_p = 0
                max_prob_prev_state = "NA"
                for previous_state in sequence_prob[layer - 1]:
                    transition = (previous_state, current_state)

                    if (input_sequence[layer - 1], current_state) in emission_dict.keys() and transition in transition_dict.keys():
                        p = sequence_prob[layer - 1][previous_state].get_max_prob() * \
                            transition_dict[(transition)] * \
                            emission_dict[(input_sequence[layer - 1], current_state)]
                    elif transition in transition_dict.keys():
                        p = sequence_prob[layer - 1][previous_state].get_max_prob() * \
                            transition_dict[(transition)] * \
                            0.00000000000001 # to allow initial state "NA" to be updated
                    else:
                        p = sequence_prob[layer - 1][previous_state].get_max_prob() * \
                            0.00000000000001 * \
                            0.00000000000001 # to allow initial state "NA" to be updated
                    sequence_prob[layer][current_state].try_add(p,previous_state)

        # backtracking to find argmax
        current_layer = n
        reverse_path = ["STOP"]
        while current_layer >= 0:
            reverse_path.append(sequence_prob[current_layer + 1][reverse_path[len(reverse_path) - 1]].get_state(self.k))
            # just means taking the current state being backtracked, find its most probable previous state as argmax
            current_layer -= 1

        predicted_sequence = reverse_path[::-1][1:len(reverse_path)-1]

        return predicted_sequence

    def set_em_params(self,params):
        """where params is a dictionary of emission params (from part2)"""
        self.em_params = params

    def viterbi(self):
        """
        iteratively runs viterbi algortihm for all input sequences,
        returns nested list of predicted state sequence for all sentences within dev-in
        """
        if self.action == "eval":
            try:
                self.tr_params = self.load_pickle("tr_params")
            except:
                self.__est_transition_params()
            try:
                self.viterbi_seq = self.load_pickle("viterbi_highest_K")
                return self.viterbi_seq
            except:
                pass
        self.__format_testdata()
        emission_dict = self.em_params
        transition_dict = self.p_y1_given_y0
        input_sequences = self.input_sequences
        pred_state_sequences = [[]]
        i = 0
        for input_sequence in input_sequences:
            for state in self.__mini_viterbi(input_sequence, emission_dict, transition_dict):
                pred_state_sequences[i].append(state)
            pred_state_sequences.append([]) # to store state sequence for next sentence
            i += 1
        pred_state_sequences.pop() # remove last []
        return pred_state_sequences

    def write_sequences(self):
        """Writes the generated sequences to dev.p3.out"""
        self.viterbi()
        f = open(self.path + "/dev.p4.out","w", encoding="utf-8")
        seq_num = 0
        word_num = 0
        for x in self.test_data:
            if len(x)<1:
                f.write("\n")
                word_num = 0
                seq_num+=1
            else:
                y = self.viterbi_seq[seq_num][word_num]
                f.write("{} {}\n".format(x,y))
                word_num+=1
        f.close()
    
    def load_pickle(self, name):
        """Loads pickle with name: 'name + path'. Returns object."""
        return pickle.load(open(name+self.path + ".p","rb"))


class queue:
    """Class to take in values of probability and state, and decide which are the top k ones"""
    def __init__(self,k, start = False):
        """K sets the 'top - k' part of the thing. start = True designates it as a starting node"""
        if start:
            self.queue = [{"p": 1.0, "previous": "NA"}]
        else:
            self.queue = [] #inside: {"p":prob,"previous":state}, highest at index 0
        self.min_prob = 0
        self.k = k


    def try_add(self,prob,state):
        """Implementation of the queue. First fills in the queue till it's at least = k. Then, checks if the value of prob is > lowest prob in the table.
        If it is, the method gets rid of the lowest, inserts the new value, and sorts the list."""
        if prob > self.min_prob or len(self.queue)<self.k:
            if len(self.queue)==self.k:
                del self.queue[-1]
            self.queue.append({"p":prob,"previous":state})
            self.queue = sorted(self.queue, key=lambda x:x["p"], reverse=True)
            self.min_prob = self.queue[-1]["p"]
        

    def get_state(self, kth):
        """Returns the kth most likely state"""
        try:
            return self.queue[kth-1]["previous"]
        except:
            return self.queue[-1]["previous"]

    def get_max_prob(self):
        """Returns the most likely state"""
        return self.queue[0]["p"]

    def __repr__(self):
        return str(self.queue)

"""Testing"""
if __name__ == "__main__":
    queues = {}
    for j in range(5):
        queues[j] = queue(3)
        for i in range(10):
            prob = random.randint(1,100)/100
            state = i
            queues[j].try_add(prob,state)

    print( queues)
    

    