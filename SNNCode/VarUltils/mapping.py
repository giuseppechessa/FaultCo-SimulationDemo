import torch
import snntorch as snn  # Ensure this module is correctly imported
from dataclasses import dataclass, field
from typing import List
from collections import defaultdict

@dataclass
class Destination:
    destination_core: int
    destination_neurons: List[int] = field(default_factory=list)


@dataclass
class NeuronConnection:
    source_neuron: int
    source_core: int
    destinations: List[Destination] = field(default_factory=list)

@dataclass
class NeuronCoreMulticast:
    source_neuron: int
    source_core: int
    destination_cores: List[int] = field(default_factory=list)

# Core–Core Multicast by neuron-core multicasts
@dataclass
class CoreCoreMulticast:
    source_core: int
    destination_cores: List[int] = field(default_factory=list)


class Mapping:
    def __init__(self, net,nir_model):
        self.net = net
        self.core_capacity = None
        

        self.mem_potential_sizes = self._get_membrane_potential_sizes()
        self.indices_to_lock = None
        self.StartingNeuronList=dict()
        self.nir_model=nir_model

    def get_total_neurons(self):
        return sum(self.mem_potential_sizes.values())

    def _get_membrane_potential_sizes(self):
        if self.net is None:
            raise ValueError("Network model has not been set. Please call set_network first.")
        
        sizes = {}
        for name, module in self.net.named_modules():
            if isinstance(module, snn.Synaptic):
                _, mem = module.init_leaky()
                sizes[name] = mem.size()[0]

            elif isinstance(module, snn.Leaky):
                mem = module.init_leaky()
                sizes[name] = mem.size()[0]

            elif isinstance(module, snn.RSynaptic):
                sizes[name] = module.linear_features

        return sizes
    
    def map_neurons(self):
        self.core_allocation, self.NIR_to_cores, self.neuron_to_core = self._allocate_neurons_to_cores()
        self.log()

    def set_core_capacity(self, cc):
        self.core_capacity = cc

    def log(self, dut=None):

        print("\n----- MAPPING -----\n")

        for layer_name, size in self.mem_potential_sizes.items():
            temp = f"Layer: {layer_name}, Number of neurons: {size}"
            if dut is not None:
                    dut._log.info(temp)
            else:
                print(temp)

        print("CORE CAPACITY", self.core_capacity)
        print("CORE ALLOCATION:",self.core_allocation)
        print("NIR TO CORES:",self.NIR_to_cores)

        
    
    def _allocate_neurons_to_cores(self):
        core_allocation = {}
        NIR_to_cores = {}
        neuron_to_core = {}

        core_id = 0
        core_start_index = 0
        current_core_neurons = 0
        full_capacity_reached = False

        layer_names = list(self.mem_potential_sizes.keys())
        last_layer_name = layer_names[-1]
        PrevSize=0
        layer_start_index=0
        for layer_name, num_neurons in self.mem_potential_sizes.items():
            self.StartingNeuronList[layer_name]=PrevSize
            PrevSize+=num_neurons
            #This restart numeration when moving from layer to layer
            layer_start_index = 0

            if layer_name == last_layer_name:
                if num_neurons > self.core_capacity:
                    raise Exception("Output layer does not fit in one core!")

                # Ensure the last layer is in the same core
                if not full_capacity_reached:
                    core_id += 1
                core_start_index = 0
                current_core_neurons = 0
                #layer_start_index = core_start_index
                layer_end_index = layer_start_index + num_neurons - 1
                core_allocation[layer_name] = [(core_id, layer_start_index, layer_end_index)]
                NIR_to_cores[layer_name] = [(core_id, layer_end_index + 1 - layer_start_index)]
                for neuron_id in range(layer_start_index, layer_end_index + 1):
                    neuron_to_core[layer_name + "-" + str(neuron_id)] = core_id
                break

            while num_neurons > 0:
                full_capacity_reached = False
                available_space = self.core_capacity - current_core_neurons
                neurons_to_allocate = min(num_neurons, available_space)

                layer_end_index = layer_start_index + neurons_to_allocate - 1

                if layer_name not in core_allocation:
                    core_allocation[layer_name] = []
                    NIR_to_cores[layer_name] = []

                core_allocation[layer_name].append((core_id, layer_start_index, layer_end_index))
                NIR_to_cores[layer_name].append((core_id, layer_end_index + 1 - layer_start_index))

                for neuron_id in range(layer_start_index, layer_end_index + 1):
                    neuron_to_core[layer_name + "-" + str(neuron_id)] = core_id

                current_core_neurons += neurons_to_allocate
                layer_start_index += neurons_to_allocate
                num_neurons -= neurons_to_allocate

                if current_core_neurons == self.core_capacity:
                    full_capacity_reached = True
                    core_id += 1
                    core_start_index = 0
                    current_core_neurons = 0
                else:
                    core_start_index = layer_start_index
        return core_allocation, NIR_to_cores, neuron_to_core

    def GetConnections(self):
        routing_matrices = {}
        all_connections: List[NeuronConnection] = []

        # counstricting routing matrices
        for layer_name, size in self.mem_potential_sizes.items(): # a way to get the layer names
            # we have two layers: lif1 and lif2

            routing_matrix = torch.zeros((size)) # routing matrix for the current layer
            for idx in range(size): # iterate over the neurons in the layer

                if layer_name in routing_matrices:
                    continue

                # unique identifier for the neuron, e.g. "lif1-0"
                routing_id = layer_name +"-"+ str(idx)
                # Get the source core for the neuron
                source_core = self.neuron_to_core[routing_id]

                if layer_name == "lif1":
                    target_edges = [(excepted_src, excepted_dst) for excepted_src, excepted_dst in self.nir_model.edges if excepted_src == "lif1.w_rec" or excepted_src == "fc2"]

                    for (src, dst) in target_edges:
                        if src in self.nir_model.nodes and self.nir_model.nodes[src].weight is not None:
                            connections = self.nir_model.nodes[src].weight.T[idx, :]  # get the connection weights for the current neuron
                        else:
                            print("No weight information available for this node.")
                            connections = np.zeros((size, ), dtype=np.float32)
                        grouped: Dict[int, List[int]] = defaultdict(list)
                        for neuron_idx, weight in enumerate(connections):
                            if weight != 0:
                                dst_layer = dst.split(".")[0]  # 获取源层名
                                dest_neuron_id = f"{dst_layer}-{neuron_idx}"
                                dest_core = self.neuron_to_core[dest_neuron_id]
                                grouped[dest_core].append(neuron_idx+self.StartingNeuronList[dst_layer])

                        # get the destination list for this source neuron
                        dest_list = [
                            Destination(destination_core=core, destination_neurons=neurons)
                            for core, neurons in grouped.items()
                        ]

                        conn = NeuronConnection(
                            source_neuron=routing_id,
                            source_core=source_core,
                            destinations=dest_list
                        )
                        all_connections.append(conn)
        return all_connections
