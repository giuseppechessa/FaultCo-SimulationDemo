import torch

class Variables(object):
    def __init__(self):
        self.num_inputs = 20
        self.num_hidden1 = 100
        self.num_outputs = 2
        self.num_cores = 6

        self.recall_duration = 20
        self.t_cue_spacing = 20
        self.silence_duration = 30
        self.n_cues = 9
        self.t_cue = 15
        self.p_group = 0.7
        self.num_steps = int(self.t_cue_spacing * self.n_cues + self.silence_duration + self.recall_duration)
