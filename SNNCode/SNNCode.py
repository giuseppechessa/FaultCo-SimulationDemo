#!/usr/bin/env python

import torch
import snntorch as snn

import os
import matplotlib.pyplot as plt
import numpy as np
import math
import copy
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
import sqlite3
import sys
import re
import argparse
import subprocess
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import  statistics


from VarUltils.graph import Graph
from VarUltils.mapping import Mapping, Destination,NeuronConnection,NeuronCoreMulticast,CoreCoreMulticast
from VarUltils.dataset import BinaryNavigationDataset
from VarUltils import utils
from VarUltils.optionsBN import Variables
from VarUltils.cam import SimpleCAMGenerator, CAMEntry

from BinaryModel.model import SpikingNet
import EndToEndInference.EndtoEndBN as EtE


def reset_spike_record_and_hooks():
    global spike_record, hooks

    # Clear the spike_record dictionary
    spike_record = {}

    # Remove existing hooks if they are already registered
    if 'hooks' in globals():
        for hook in hooks:
            hook.remove()
            hooks = []

# Function to create a hook that records spikes
def create_spike_hook(layer_name):
    def hook(module, input, output):
        if layer_name not in spike_record:
            spike_record[layer_name] = []
        spike_record[layer_name].append(output[0].detach().cpu()) # 0 index refers to spk
    return hook



def plot_spike_raster(spike_tensor, title='Spike Raster Plot'):
    """
    spike_tensor: torch.Tensor or numpy.ndarray of shape (T, N)
    T: time steps, N: neurons
    """
    if isinstance(spike_tensor, torch.Tensor):
        spike_tensor = spike_tensor.numpy()

    T, N = spike_tensor.shape
    fig, ax = plt.subplots(figsize=(10, 4))

    for neuron_idx in range(N):
        spike_times = np.where(spike_tensor[:, neuron_idx] > 0)[0]
        ax.scatter(spike_times, [neuron_idx] * len(spike_times), s=2, color='black')

    ax.set_xlabel('Time step')
    ax.set_ylabel('Neuron index')
    ax.set_title(title)
    ax.set_ylim([-1, N + 1])
    ax.set_xlim([0, T])
    plt.tight_layout()
    plt.show()



# Build comprehensive neuron lookup tables for online package generation
@dataclass
class NeuronLookup:
    """Complete neuron lookup information for online package generation"""
    neuron_id: str  # e.g., "lif1-0"
    layer_name: str  # e.g., "lif1"
    local_index: int  # index within layer
    global_index: int  # unique index across all neurons
    source_core: int  # which core this neuron belongs to
    destination_cores: List[int]  # which cores this neuron sends to

def build_neuron_lookup_table(mapping, nc_multicasts) -> Dict[str, NeuronLookup]:
    """
    Build comprehensive lookup table for all neurons
    Returns: Dictionary mapping neuron_id to NeuronLookup
    """
    lookup_table = {}
    global_index = 0

    # Build destination cores mapping from nc_multicasts
    dest_cores_map = {}
    for multicast in nc_multicasts:
        neuron_id = multicast.source_neuron
        if neuron_id not in dest_cores_map:
            dest_cores_map[neuron_id] = []
        dest_cores_map[neuron_id].extend(multicast.destination_cores)

    # Remove duplicates and sort
    for neuron_id in dest_cores_map:
        dest_cores_map[neuron_id] = sorted(list(set(dest_cores_map[neuron_id])))

    for layer_name, size in mapping.mem_potential_sizes.items():
        for local_idx in range(size):
            neuron_id = f"{layer_name}-{local_idx}"
            source_core = mapping.neuron_to_core[neuron_id]
            destination_cores = dest_cores_map.get(neuron_id, [])

            lookup_table[neuron_id] = NeuronLookup(
                neuron_id=neuron_id,
                layer_name=layer_name,
                local_index=local_idx,
                global_index=global_index,
                source_core=source_core,
                destination_cores=destination_cores
            )
            global_index += 1

    return lookup_table

def build_ID_lookup(mapping) -> Dict[int,str]:
    IdLookup=dict()
    for layer_name, size in mapping.mem_potential_sizes.items():
        print(layer_name,mapping.StartingNeuronList[layer_name])
        for local_idx in range(size):
            IdLookup[local_idx+mapping.StartingNeuronList[layer_name]]=neuron_id = f"{layer_name}-{local_idx}"

    return IdLookup



def regexp(expr, item):
    reg = re.compile(expr)
    return reg.search(item) is not None

@torch.no_grad()
def evaluate(OPTION,numErrori):
    net_copy.eval()
    net_copy.to(device)

    total_correct, total_samples = 0, 0

    all_predictions = []
    all_labels = []
    all_counts = []

    # all_spikes = []
    i=0
    i=0
    if(OPTION!=None):
        conFaultPos = sqlite3.connect("./database/FaultInjPositions.sql")
        conFaultPos.create_function("REGEXP", 2, regexp)
        curFaultPos = conFaultPos.cursor()
        Command = f"SELECT * FROM 'FaultPlaces' where faultPlace REGEXP ?"
        res = curFaultPos.execute(Command,[OPTION])
        ErrList=res.fetchall()
        if(len(ErrList)==0):
            print("ERROR in REGEXP try again");
            sys.exit()
        new_dict = dict();
        SetError = -1;
        new_dict[SetError]=1;
        
        con = sqlite3.connect("./FaultsFinal"+OPTION+".sql")
        cur = con.cursor()
        conFaultFinal = sqlite3.connect("./FaultsFinalSwitch.sql")
        curFaultFinal = conFaultFinal.cursor()
        #curFaultFinal.execute("CREATE TABLE Mousetrap(ID int,FaultPlace string, BitFault int,Polarity int,Sample int, PacketsDropped int, accuracy float,MISSED int)")
        conFaultFinal.create_function("REGEXP", 2, regexp)
        try:
            res=curFaultFinal.execute("Select Distinct ID from Mousetrap")
        except:
            print("NoFaultsyet")
        AlreadyAnalized=res.fetchall();

        for AlAn in AlreadyAnalized:
            new_dict[AlAn[0]]=1;
        try:
            cur.execute("DROP TABLE Mousetrap; ")
        except:
            i=0
        cur.execute("CREATE TABLE Mousetrap(ID int,FaultPlace string, BitFault int,Polarity int,Sample int, PacketsDropped int, accuracy float,MISSED int)")
        con.commit()

    for ErrorsNum in range(numErrori):
        if(OPTION!=None):
            SetError=int(random.uniform(0,len(ErrList)));
            while (ErrList[SetError][0] in new_dict):
                SetError=int(random.uniform(0,len(ErrList)));
            ErrNum=ErrList[SetError][0]
            ID=ErrList[SetError][1]
            Polarity=ErrList[SetError][2]
            Bit=ErrList[SetError][3]
            new_dict[ErrNum]=1;
            print(SetError," ",ErrNum," ",ID," ",Polarity," ",Bit)
        i=0
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Evaluating")
            for x, y in pbar:
                if(OPTION==None):
                    error_config="NoneWithExtraSteps"
                    OPTION=""
                    ErrNum=0
                else:
                    error_config="Error"
                # Clean up old databases
                for db_path in ["./database/DataSx"+OPTION+".sql", "./database/DataRx"+OPTION+".sql"]:
                    if os.path.exists(db_path):
                        os.remove(db_path)
                if error_config==None:
                    boh=1;
                elif error_config == "NoneWithExtraSteps":
                    subprocess.Popen(["./SWITCH_SIM5V"])
                else:
                    subprocess.Popen(["./SWITCH_SIM5V", "-C",OPTION,str(ErrNum)])
                same_step_inference = EtE.EndToEndHardwareInference(
                    network=net_copy,
                    neuron_lookup=neuron_lookup,
                    mapping=mapping,
                    all_connections=all_connections,
                    device=device,
                    dataSx_path="./database/DataSx"+OPTION+".sql", 
                    dataRx_path="./database/DataRx"+OPTION+".sql",
                    IdLookup=IdLookup
                )

                x = x.to(torch.float32).to(device)
                y = y.to(torch.long).to(device)

                # delete batch dimension
                x = x.squeeze(0)  # [B, T, C] -> [T, C]
                
                spk_out, final_states, stats,WrongSpikes,Missed = same_step_inference.end_to_end_inference(
                    input_sequence=x,
                    max_steps=None,
                    hardware_wait_time=0,
                    keep_database=False, 
                    error_config=error_config,
                    OPTION=OPTION
                )

                
                predictions = spk_out.sum(dim=0).squeeze(0).argmax(dim=0)
                CorrLabel = y.cpu().item();
                if(predictions==CorrLabel):
                    ACC=1
                else :
                    ACC = 0

                if(OPTION!=""):
                    Command = f"INSERT INTO Mousetrap VALUES({ErrNum},\'{ID}\',{Bit},{Polarity},{i},{WrongSpikes},{ACC},{Missed})"
                    print(Command)
                    cur.execute(Command)
                    con.commit()
                i=i+1
                all_predictions.append(predictions.cpu().item())
                all_counts.append(abs(spk_out.sum(dim=0).squeeze(0)[0] - spk_out.sum(dim=0).squeeze(0)[1]))
                all_labels.append(y.cpu().item())

                if len(all_predictions) > 0:
                    current_acc = accuracy_score(all_labels, all_predictions)
                    current_precision = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
                    current_recall = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
                    current_f1 = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
                    current_count =  statistics.fmean(all_counts, weights=None);

                pbar.set_postfix({
                    'Accuracy': f'{current_acc:.4f}',
                    'Precision': f'{current_precision:.4f}',
                    'Recall': f'{current_recall:.4f}',
                    'F1_Score': f'{current_f1:.4f}',
                    'DiffCount': f'{current_count:.4f}',
                    'Samples': len(all_predictions)
                })

            # Cleanup
            same_step_inference.cleanup()

    conFaultPos.close();


parser = argparse.ArgumentParser()
parser.add_argument("-num", required=True,help="Number of errors")
parser.add_argument("-T", required=False,help="Target regexp")
args = parser.parse_args()
print(args)

# ## Device
if torch.backends.mps.is_available():
    device = torch.device("mps") # For Apple Silicon Macs
elif torch.cuda.is_available():
    device = torch.device("cuda") # For NVIDIA GPUs
else:
    device = torch.device("cpu")

print("Using device:", device)


# ## Initialize the sample model with sample data
v = Variables()


# # NIR

torch.manual_seed(42)
net = SpikingNet(v)
sample_data = torch.randn(v.num_steps, v.num_inputs)


# Initialize the network with sample data
net.load_state_dict(torch.load("./model/best_rsnn_rl_nav_8.pt",map_location=device))
net = utils.init_network(net, sample_data)


# -------------------------------------------------
# Get the NIR graphv
gp = Graph(net, v.num_inputs)

# Get the NIR model
nir_model = gp.nir_model


# # Core-Core Multicast Tree

# Find all matching edges
target_edges = [(src, dst) for src, dst in nir_model.edges if src == "lif1.w_rec" or src == "fc2"]
# target_edges = target_edges[0] if target_edges else None



# Map neurons to cores
mapping = Mapping(net,nir_model)
total_neurons = mapping.get_total_neurons()
core_capacity = max(math.ceil((total_neurons - v.num_outputs) / (v.num_cores - 1)), v.num_outputs)
mapping.set_core_capacity(core_capacity)
mapping.map_neurons()







all_connections =mapping.GetConnections()

nc_multicasts: List[NeuronCoreMulticast] = []

for conn in all_connections:
    nc_multicasts.append(
        NeuronCoreMulticast(
            source_neuron=conn.source_neuron,
            source_core=conn.source_core,
            destination_cores=[
                dest.destination_core
                for dest in conn.destinations
            ]
        )
    )

# group the neuron-core multicasts by source_core
_grouped = defaultdict(set)  # use set to deduplicate
for m in nc_multicasts:
    _grouped[m.source_core].update(m.destination_cores)

# build CoreCoreMulticast objects
cc_multicasts: List[CoreCoreMulticast] = [
    CoreCoreMulticast(source_core=src, destination_cores=sorted(list(dests)))
    for src, dests in _grouped.items()
]

# print the Core–Core multicasts
for cc in cc_multicasts:
    print(f"Core {cc.source_core}  ->  Cores {cc.destination_cores}")


# # Create dataset and dataloader
val_set = torch.load('data/test_set_8.pt', weights_only=False)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=True, num_workers=4, prefetch_factor=2)

# ## Visialization Dataset
print("One batch data shape:", next(iter(val_loader))[1].shape) # (B, T, imputs)
print("One batch label shape:", next(iter(val_loader))[1].shape) # (B)


def VisualizeSpikes(val_loader):
    # Visualize the data
    for spk_data, target_data in val_loader:
        data_visual = spk_data[0]
        print(spk_data.size())

        fig = plt.figure(facecolor="w", figsize=(10, 5))
        ax = fig.add_subplot(111)
        ax.set_yticks(np.arange(0, 40, 2)) 
        snn.spikeplot.raster(data_visual, ax, s=5, c="blue")

        plt.title("Input Sample")
        plt.xlabel("Time step")
        plt.ylabel("Neuron Number")
        plt.savefig("./Spikes.pdf")
        #plt.show()
        break  # Only display the first batch

VisualizeSpikes(val_loader)
# ## Evaluate the model

# ## Get Spikes
# 
# Here we use the hook tool to extract the spikes in the model. Hook is a automatical tool to collect the spikes inside the model after each forward step. But in the later networking hacking process, we whould directly generate packages by the spikes inside the model instead of using hook. The new method is easy to understand and maintain, and is more free to modify the code.

net_copy = copy.deepcopy(net).to(device)

# Dictionary to store spikes from each layer
spike_record = {}
hooks = []


reset_spike_record_and_hooks()
# Attach hooks automatically to all Leaky layers
for name, module in net_copy.named_modules():
    if isinstance(module, snn.Leaky) or isinstance(module, snn.RSynaptic):
        hooks.append(module.register_forward_hook(create_spike_hook(name)))

# Get a sample input
data, _ = val_set[0]

# Record spikes
_, _ = net_copy(data.to(device))

# Convert spike records to tensors for easier analysis
for layer_name in spike_record:
    spike_record[layer_name] = torch.squeeze(torch.stack(spike_record[layer_name]))



# Count the number of spikes in each layer
spike_counts = {layer_name: spikes.sum().item() for layer_name, spikes in spike_record.items()}
# -------------------------------------------------

if 'hooks' in globals():
    for hook in hooks:
        hook.remove()
        hooks = []


# Build the lookup table
neuron_lookup = build_neuron_lookup_table(mapping, nc_multicasts)
IdLookup=build_ID_lookup(mapping)



# ## CAM Table
# Generate Instance（use neuron_lookup abd all_connections to generate CAM）
simple_cam_generator = SimpleCAMGenerator(neuron_lookup, all_connections)
# Generate CAM tables and display them
print("Generating CAM tables with filtering stats...")
cam_tables = simple_cam_generator.display_cam_tables_with_filtering_stats()

# Save to database
print("\nSaving CAM tables to database...")
db_path = simple_cam_generator.save_cam_tables_to_database(cam_tables, "./database/CAMTables.sql")

if db_path:
    print(f"CAM tables successfully saved to: {db_path}")
else:
    print("Failed to save CAM tables to database")




# test_acc = evaluate()
evaluate(args.T,int(args.num))
# print(f"Test accuracy: {test_acc:.4f}")




