import torch
import numpy as np
import EndToEndInference.PackageGenerator as PG

class EndToEndHardwareInference:
    """
    Make the single step inference process with the complete pipeline
    """

    def __init__(self, network, neuron_lookup, mapping, all_connections,
                 dataSx_path="./database/DataSx.sql", 
                 dataRx_path="./database/DataRx.sql",
                 IdLookup=None,
                 device="cpu"):
        self.network = network
        self.neuron_lookup = neuron_lookup
        self.mapping = mapping 
        self.all_connections = all_connections
        self.device = device
        self.IdLookup=IdLookup

        self.package_generator = PG.PackageGenerator(
            neuron_lookup=neuron_lookup,
            network_reference=network,  # 传入网络引用
            dataSx_path=dataSx_path,
            dataRx_path=dataRx_path,
            IdLookup=IdLookup
        )

        nonzero_recurrent,nonzero_feedforward = self.package_generator.getNonzeroMatrix();

        # # TODO: Can use PackageDecoder to decode more complex packages in the future
        # self.package_decoder = PackageDecoder(
        #     neuron_lookup=neuron_lookup,
        #     all_connections=all_connections
        # )

        self.state_corrector = PG.HardwareStateCorrector(
            network=network,
            neuron_lookup=neuron_lookup,
            all_connections=all_connections,
            package_decoder=None,
            IdLookup=IdLookup,
            nonzero_recurrent=nonzero_recurrent,
            nonzero_feedforward=nonzero_feedforward
        )

        # init
        self.network_states = self.init_network_states()

        self.stats = {
            'total_steps': 0,
            'correction_steps': 0,
            'packages_sent': 0,
            'packages_received': 0
        }
        self.lif1Matrix=torch.cat((self.network.lif1.recurrent.weight.data.cpu(),self.network.fc2.weight.data.cpu()))



    def init_network_states(self):
        spk1, syn1, mem1 = self.network.lif1.init_rsynaptic()
        mem2 = self.network.lif2.init_leaky()
        return {
            'spk1': spk1,
            'syn1': syn1,
            'mem1': mem1,
            'mem2': mem2
        }

    def get_network_states(self):
        return self.network_states

    def update_network_states(self, spk1, syn1, mem1, mem2):
        self.network_states = {
            'spk1': spk1,
            'syn1': syn1,
            'mem1': mem1,
            'mem2': mem2
        }

    def single_step_with_same_time_correction(self, input_data, hardware_wait_time=0.1, keep_database=False, error_config=None, WrongSpikes=0, Missed = 0,OPTION=None):
        """
        Pipeline:
        1. Execute lif1 inference to obtain spk1 and updated syn1
        2. Actively call generate_packages(spk1) to generate packages
        3. Receive packages returned by hardware
        4. Calculate correction parameters and directly correct syn1 and cur2 for the current time step
        5. Continue executing fc2 and lif2 to complete inference

        Args:
            input_data: (batch_size, input_size)
            hardware_wait_time: delay time
            keep_database: keep the database or not

        Returns:
            output: output spikes of the network
            states: net states
            correction_applied: whether correction is applied
        """

        # print(f"\\n=== Time Step {self.stats['total_steps']} === [Same-Step Correction + New PackageGenerator]")
        # print(f"Database Model: {'cumulated' if keep_database else 'cleaned'}")

        # Get the current network status
        current_states = self.get_network_states()
        x = input_data
        spk1 = current_states['spk1']
        syn1 = current_states['syn1'] 
        mem1 = current_states['mem1']
        mem2 = current_states['mem2']

        # Input layer
        cur1 = self.network.fc1(x)
        # Recurrent layer
        spk1, syn1, mem1 = self.network.lif1(cur1, spk1, syn1, mem1)
        cur2 = self.network.fc2(spk1)

        # Generate packages based on spk1
        self.package_generator.generate_packages(spk1,'lif1')
        # get the current package
        transmitted_packages = self.package_generator.get_current_transmitted_packages()
        self_communication_connections = self.package_generator.get_current_self_communication_connections()

        self.stats['packages_sent'] += len(transmitted_packages)
        total_self_comm = sum(len(connections) for connections in self_communication_connections.values())
        correction_applied = False
        if(error_config!=None):
            ################################## hardware simulation###############################
            pipe_path = "./database/myfifo"+OPTION
            with open(pipe_path, "w") as pipe:#Send Signal to Start Hardware simulation
                pipe.write("1")
            with open(pipe_path, "r") as pipe: 
                Missed = pipe.read() #Waiting End of Hardware Simulation
            ################################## hardware simulation###############################
            Missed=int(Missed[:-1])
            #print(Missed)
            #Missed=0
            hardware_packages = self.package_generator.read_packages_from_dataRx()
            self.stats['packages_received'] += len(hardware_packages)

            
            current_predictions = {
                'lif1': spk1.detach().clone(),
            }

            # time_start = time.time()
            cur_sub, cur_add = self.state_corrector.generate_corrections_from_packages(
                current_predictions, 
                hardware_packages,
                self_communication_connections
            )
            correction_count = (len(cur_sub) if cur_sub else 0) + (len(cur_add) if cur_add else 0)
        else:
            cur_sub = [];
            cur_add = [];

        # Apply corrections to syn1 and cur2
        # time_start = time.time()
        WrongSpikes = WrongSpikes+len(cur_sub) + len(cur_add); 
        
        if len(cur_sub) != 0 or len(cur_add) != 0:
            for layer_name, size in self.mapping.mem_potential_sizes.items():
               if(layer_name=='lif1'):
                   recurrent_weights = self.lif1Matrix
                   syn,cur=syn1,cur2
                   alpha=1/self.network.lif1.alpha
                   #print(alpha)

               if cur_sub:
                   start = self.mapping.StartingNeuronList[layer_name]
                   end = start + size

                   arr = np.array(cur_sub)  # shape (N, 3)
                   multipliers = arr[:, 0]
                   neuron_idx = arr[:, 1]
                   src_idx = arr[:, 2]

                   mask = (neuron_idx >= start) & (neuron_idx < end)
                   multipliers = multipliers[mask]
                   neuron_idx = neuron_idx[mask]
                   src_idx = src_idx[mask]

                   rows = src_idx - start
                   cols = neuron_idx - start
                   weights = recurrent_weights[rows, cols]
                   updates = weights * multipliers

                   mask_cur = rows >= size
                   mask_syn = ~mask_cur

                   if np.any(mask_syn):
                        print(updates[mask_syn])
                        print(updates[mask_syn]*alpha)
                        syn[ :, rows[mask_syn]] -= updates[mask_syn]*alpha

                   if np.any(mask_cur):
                       cur[ :, rows[mask_cur] - size] -= updates[mask_cur]
               if cur_add:
                   start = self.mapping.StartingNeuronList[layer_name]
                   end = start + size

                   arr = np.array(cur_add)  # shape (N, 3)
                   multipliers = arr[:, 0]
                   neuron_idx = arr[:, 1]
                   src_idx = arr[:, 2]

                   mask = (neuron_idx >= start) & (neuron_idx < end)
                   multipliers = multipliers[mask]
                   neuron_idx = neuron_idx[mask]
                   src_idx = src_idx[mask]

                   rows = src_idx - start
                   cols = neuron_idx - start
                   weights = recurrent_weights[rows, cols]
                   updates = weights * multipliers

                   mask_cur = rows >= size
                   mask_syn = ~mask_cur

                   if np.any(mask_syn):
                       syn[ :, rows[mask_syn]] += updates[mask_syn]*alpha

                   if np.any(mask_cur):
                       cur[ :, rows[mask_cur] - size] += updates[mask_cur]

               # update` network state
               if(layer_name=='lif1'):
                   syn1,cur2 = syn,cur

        # lif2 inference
        output, mem2 = self.network.lif2(cur2, mem2)

        # Update network states
        self.update_network_states(spk1, syn1, mem1, mem2)
        self.package_generator.step(keep_database=keep_database)
        self.stats['total_steps'] += 1

        if correction_applied:
            self.stats['correction_steps'] += 1
        final_states = self.get_network_states()
        return output, final_states, correction_applied,WrongSpikes,Missed




    def end_to_end_inference(self, input_sequence, max_steps=None, hardware_wait_time=0.1, keep_database=False, error_config=None,OPTION=None):

        if max_steps is None:
            max_steps = len(input_sequence)
        else:
            max_steps = min(max_steps, len(input_sequence))

        outputs = []
        WrongSpikes=0
        Missed=0

        for t in range(max_steps):

            current_input = input_sequence[t].unsqueeze(0)

            output, states, correction_applied,WrongSpikes,Missed = self.single_step_with_same_time_correction(
                input_data=current_input,
                hardware_wait_time=hardware_wait_time,
                keep_database=keep_database,
                error_config=error_config,
                WrongSpikes=WrongSpikes,
                Missed=Missed,
                OPTION=OPTION
                )

            # output, states, correction_applied = self.single_step_with_delayed_correction(
            #     input_data=current_input,
            #     hardware_wait_time=hardware_wait_time,
            #     keep_database=keep_database,
            #     error_config=error_config
            # )

            outputs.append(output)

        if keep_database:
            try:
                final_dataSx = self.package_generator.get_current_transmitted_packages()
                final_dataRx = self.package_generator.read_packages_from_dataRx()
                # print(f"Final DataSx packages: {len(final_dataSx)}")
                # print(f"Final DataRx packages: {len(final_dataRx)}")
            except:
                print("Database state query failed")

        final_states = self.get_network_states()
        if(error_config!=None):
            pipe_path = "./database/myfifo"+OPTION
            with open(pipe_path, "w") as pipe:
                pipe.write("FINISHED")

        return torch.stack(outputs, dim=0), final_states, self.stats.copy(),WrongSpikes,Missed

    def reset_stats(self):
        self.stats = {
            'total_steps': 0,
            'correction_steps': 0,
            'packages_sent': 0,
            'packages_received': 0
        }

    def cleanup(self):
        try:
            # close databases
            self.package_generator.reset()
            # print("Databases closed")
            self.reset_stats()

        except Exception as e:
            print(f"Error: {e}")
