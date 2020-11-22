import numpy as np

# estimate transition parameters from the training set using MLE 

# count(u,v) Number of times we see a transition from u to v
# count(u) Number of times we see the state u in the training set

def est_transition_params(self):
        y_vals = self.train_data[:,1]
        count_y = {}
        count_y0_to_y1 = {}
        previous_state = ""
        self.p_y1_given_y0 = {}
        # result = str.split('\n\n')
        # result = result[:-1]
        
        # for entry in result: 
        for line in self.train_data:
            if previous_state == "": #assume empty line is in train data - TO REVIEW
                previous_state = "START"
                state = line.split()[1]
                transition = (previous_state, state)
                count_y0_to_y1[(transition)] = count_y[(transition)] + 1 if transition in count_y else 1
                count_y["START"] = count_y["START"] + 1 if "START" in count_y else 1

            elif len(line) == 0:
                state = "STOP"
                transition = (previous_state, state)
                count_y0_to_y1[(transition)] = count_y[(transition)] + 1 if transition in count_y else 1
                previous_state = ""

            else:
                state = line.split()[1]
                transition = (previous_state, state)
                count_y0_to_y1[(transition)] = count_y[(transition)] + 1 if transition in count_y else 1
                count_y[state] = count_y[state] + 1 if state in count_y else 1
                previous_state == state

        # for entry in y_vals:
        #     if entry not in(count_y):
        #         count_y[entry] = 1
        #     else:
        #         count_y[entry] +=1

        for entry, count in count_y0_to_y1.items():
            self.p_y1_given_y0[entry] = count/count_y[entry[0]]
        
        print(self.p_y1_given_y0)