import torch
import snntorch as snn
import snntorch.functional as SF
from snntorch import RSynaptic
import torch.nn as nn
from snntorch import surrogate

class SpikingNet(torch.nn.Module):
    def __init__(self, opt, spike_grad=surrogate.fast_sigmoid(), learn_alpha=True, learn_beta=True, learn_treshold=True):
        super().__init__()

        # Initialize layers
        # self.fc1 = nn.Linear(opt.num_inputs, opt.num_hidden1)
        # self.fc1.__setattr__("bias",None) # biological plausability
        self.fc1 = nn.Linear(opt.num_inputs, opt.num_hidden1, bias=False)
        self.lif1 = RSynaptic(alpha=0.9, beta=0.9, spike_grad=spike_grad, learn_alpha=True, learn_threshold=True, linear_features=opt.num_hidden1, reset_mechanism="subtract", reset_delay=False, all_to_all=True)
        # self.lif1.recurrent.__setattr__("bias",None) # biological plausability
        if hasattr(self.lif1, 'recurrent') and hasattr(self.lif1.recurrent, 'bias'):
            self.lif1.recurrent.bias = None

        # self.fc2 = nn.Linear(opt.num_hidden1, opt.num_outputs)
        # self.fc2.__setattr__("bias",None) # biological plausability
        self.fc2 = nn.Linear(opt.num_hidden1, opt.num_outputs, bias=False)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=spike_grad)

        self.num_steps = opt.num_steps

    def init_neurons():
        pass

    def forward_one_ts(self, x, spk1, syn1, mem1, mem2, cur_sub=None, cur_add=None, time_first=True):

        if not time_first:
            #test = data
            x=x.transpose(1, 0)
        curr_sub_rec = []
        curr_add_rec = []
        
        curr_sub_fc = []
        curr_add_fc = []
        if cur_sub is not None:
            for element in cur_sub:
                if element[2] > 99:
                    curr_sub_fc.append(element)
                    pass
                else:
                    curr_sub_rec.append(element)
                    pass

        if cur_add is not None:
            for element in cur_add:
                if element[2] > 99:
                    curr_add_fc.append(element)
                    pass
                else:
                    curr_add_rec.append(element)
                    pass

        ## Input layer
        cur1 = self.fc1(x)

        ### Recurrent layer - Manual calculation with corrections
        # 1. Calculate original recurrent connection output
        recurrent_output = self.lif1.recurrent(spk1)
        
        # 2. Apply corrections to recurrent output
        for element in curr_sub_rec:
            multiplier = element[0]
            w_idx = (element[2], element[1])
            cur_idx = element[2]
            weight = self.lif1.recurrent.weight.data[w_idx].item()
            recurrent_output[:, cur_idx] -= weight * multiplier
            
        for element in curr_add_rec:
            multiplier = element[0] 
            w_idx = (element[2], element[1])
            cur_idx = element[2]
            weight = self.lif1.recurrent.weight.data[w_idx].item()
            recurrent_output[:, cur_idx] += weight * multiplier
        
        # 3. Calculate delayed reset based on OLD membrane potential (like RSynaptic)
        delayed_reset = self.lif1.mem_reset(mem1)
        
        # 4. Manually calculate syn using corrected recurrent output
        syn1 = self.lif1.alpha.clamp(0, 1) * syn1 + cur1 + recurrent_output
        
        # 5. Manually calculate mem and apply delayed reset (like _base_sub)
        mem1 = self.lif1.beta.clamp(0, 1) * mem1 + syn1
        if self.lif1.reset_mechanism_val == 0:  # subtract
            mem1 -= delayed_reset * self.lif1.threshold
        elif self.lif1.reset_mechanism_val == 1:  # zero
            # For zero reset mechanism, need to handle properly
            mem1_no_reset = self.lif1.beta.clamp(0, 1) * mem1 + syn1
            mem1 = mem1_no_reset * (1 - delayed_reset)
        
        # 6. Calculate spike
        spk1 = self.lif1.fire(mem1)
        
        # 7. Apply immediate reset (since reset_delay=False)
        # This avoids double reset by only resetting newly fired spikes
        immediate_reset = spk1 - delayed_reset
        if self.lif1.reset_mechanism_val == 0:  # subtract
            mem1 = mem1 - immediate_reset * self.lif1.threshold
        elif self.lif1.reset_mechanism_val == 1:  # zero
            mem1 = mem1 - immediate_reset * mem1
        
        # Update lif1 internal states for next time step
        self.lif1.spk = spk1
        self.lif1.syn = syn1
        self.lif1.mem = mem1
        
        # 7. Manually trigger hooks for package generation
        # Since we bypassed the normal forward() call, we need to trigger hooks manually
        if hasattr(self.lif1, '_forward_hooks') and self.lif1._forward_hooks:
            # Create fake input/output for hook compatibility
            fake_input = (cur1, spk1, syn1, mem1)
            fake_output = (spk1, syn1, mem1)
            
            for hook in self.lif1._forward_hooks.values():
                hook(self.lif1, fake_input, fake_output)

        ### Output layer
        cur2 = self.fc2(spk1)

        for element in curr_sub_fc:
            multiplier = element[0]
            w_idx = (element[2]-100, element[1])
            cur_idx = element[2]-100
            #print("WEIGHT DIMS", self.fc2.weight.data.shape)
            weight = self.fc2.weight.data[w_idx].item()

            cur2[:, cur_idx] = cur2[:, cur_idx] - weight*multiplier

        for element in curr_add_fc:
            multiplier = element[0]
            w_idx = (element[2]-100, element[1])
            cur_idx = element[2]-100

            weight = self.fc2.weight.data[w_idx].item()

            cur2[:, cur_idx] = cur2[:, cur_idx] + weight*multiplier

        spk2, mem2 = self.lif2(cur2, mem2)

        return spk2, spk1, syn1, mem1, mem2

    def forward_single_step(self, x, spk1, syn1, mem1, mem2):
        """
        Simplified single-step forward pass for online inference
        
        Args:
            x: Input for current time step (batch_size, input_size)
            spk1, syn1, mem1: Previous states of hidden layer
            mem2: Previous state of output layer
            
        Returns:
            spk2: Output spikes (batch_size, output_size)
            spk1, syn1, mem1, mem2: Updated states
        """
        # Input layer
        cur1 = self.fc1(x)
        
        # Recurrent layer
        spk1, syn1, mem1 = self.lif1(cur1, spk1, syn1, mem1)
        
        # Output layer
        cur2 = self.fc2(spk1)
        spk2, mem2 = self.lif2(cur2, mem2)
        
        return spk2, spk1, syn1, mem1, mem2

    def forward(self, x, time_first=True):

        spk1, syn1, mem1 = self.lif1.init_rsynaptic()
        mem2 = self.lif2.init_leaky()

        # Record the spikes from the hidden layer (if needed)
        spk1_rec = [] # not necessarily needed for inference
        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        if not time_first:
            #test = data
            x=x.transpose(1, 0)

        # Print the shape of the new tensor to verify the dimensions are swapped
        # print(x.shape)
        # for step in range(self.num_steps):
        for step in range(x.shape[0]):
            ## Input layer
            cur1 = self.fc1(x[step])

            ### Recurrent layer
            spk1, syn1, mem1 = self.lif1(cur1, spk1, syn1, mem1)

            ### Output layer
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)