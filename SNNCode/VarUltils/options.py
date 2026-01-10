import torch

class Variables(object):
    def __init__(self):
        self.num_inputs = 700
        self.num_hidden1 = 320
        self.num_hidden2 = 160
        self.num_hidden3 = 64
        self.num_outputs = 20
        self.core_capacity = 25 # calculated automatically during mapping
        self.num_epochs = 20
        self.lr = 1e-4
        self.target_fr = 1.0
        self.bs = 32
        self.num_cores = 6
        self.target_sparcity = 1.0
        self.wandb_key = "d1dffe837cd949cf57cea480936502435ab5a5ca"

        self.train = False

        self.recall_duration = 20
        self.t_cue_spacing = 20
        self.silence_duration = 30
        self.n_cues = 9
        self.t_cue = 15
        self.p_group = 0.7
        #self.num_steps = int(self.t_cue_spacing * self.n_cues + self.silence_duration + self.recall_duration)
        self.num_steps=120
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

class Specs(object):
    def __init__(self):
        self.ADDR_W = 5
        self.MSG_W = 10
        self.EAST, self.NORTH, self.WEST, self.SOUTH, self.L1 = range(5)
        self.NUM_PACKETS_P_INJ = 20

        self.SID    = 0b000001
        self.SID1   = 0b100000
        self.E_MASK = 0b010000
        self.N_MASK = 0b001000
        self.W_MASK = 0b000100
        self.S_MASK = 0b000010
