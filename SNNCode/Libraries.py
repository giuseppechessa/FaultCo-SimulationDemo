#!/usr/bin/env python

import torch
import snntorch as snn

import os
import matplotlib.pyplot as plt
import numpy as np
import math
import copy
import random
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

def VisualizeSpikes(val_loader,Name):
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
        plt.savefig(Name)
        #plt.show()
        break  # Only display the first batch



def regexp(expr, item):
    reg = re.compile(expr)
    return reg.search(item) is not None

@torch.no_grad()
def evaluate(net,device,val_loader,mapping,all_connections,IdLookup,neuron_lookup,OPTION,numErrori,Name):
    TotalAcc=[]
    TotalWrongSpikes=[]
    if(OPTION!=None):
        conFaultPos = sqlite3.connect("./database/FaultInjPositions.sql")
        conFaultPos.create_function("REGEXP", 2, regexp)
        curFaultPos = conFaultPos.cursor()
        res = curFaultPos.execute(f"SELECT * FROM 'FaultPlaces' where faultPlace REGEXP ?",[OPTION])
        ErrList=res.fetchall()
        if(len(ErrList)==0):
            print("ERROR in REGEXP try again");
            sys.exit()
        new_dict = dict();
        SetError = -1;
        new_dict[SetError]=1

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
        else:
            ErrNum=0
            ID=None
            Polarity=0
            Bit=0
        i=0
        
        with torch.no_grad():
            #pbar = tqdm(val_loader, desc="Evaluating")
            for x, y in val_loader:
                i+=1
                if(i<7):
                    continue
                if(OPTION==None):
                    error_config="NoneWithExtraSteps"
                    SPACER=""
                    ErrNum=0
                else:
                    error_config="Error"
                    SPACER=OPTION

                # Clean up old databases
                for db_path in ["./database/DataSx"+SPACER+".sql", "./database/DataRx"+SPACER+".sql"]:
                    if os.path.exists(db_path):
                        os.remove(db_path)

                if error_config==None:
                    pass;
                elif error_config == "NoneWithExtraSteps":
                    subprocess.Popen(["./SWITCH_SIM5V"])
                else:
                    subprocess.Popen(["./SWITCH_SIM5V", "-C",SPACER,str(ErrNum)])

                same_step_inference = EtE.EndToEndHardwareInference(
                    network=net,
                    neuron_lookup=neuron_lookup,
                    mapping=mapping,
                    all_connections=all_connections,
                    device=device,
                    dataSx_path="./database/DataSx"+SPACER+".sql", 
                    dataRx_path="./database/DataRx"+SPACER+".sql",
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
                    OPTION=SPACER
                )

                
                predictions = spk_out.sum(dim=0).squeeze(0).argmax(dim=0)
                CorrLabel = y.cpu().item();

                if(predictions==CorrLabel):
                    ACC=1
                else :
                    ACC = 0
                Command = f"INSERT INTO Mousetrap VALUES({ErrNum},\'{ID}\',{Bit},{Polarity},{0},{WrongSpikes},{ACC},{Missed})"
                print(f"ERRID:{ErrNum}, at fault location:{ID}-{Bit}, with polarity:{Polarity}: WrongSpikes:{WrongSpikes}, Accuracy:{ACC}")
                TotalAcc.append(ACC)
                TotalWrongSpikes.append(WrongSpikes)
                i=i+1
                break


            # Cleanup
            same_step_inference.cleanup()
    if(OPTION!=None):
        conFaultPos.close();
    if(Name=="./Mapping2FaultsInj.sql"):
        Tot=890356
    else:
        Tot=890356
    accmean=statistics.mean(TotalAcc)*100
    wrongmean=(statistics.mean(TotalWrongSpikes)/Tot)*100
    print(accmean,wrongmean)
    con = sqlite3.connect(Name)
    cur = con.cursor()
    try:
        cur.execute("CREATE TABLE FaultInjResults(FunctionalBlock string,PacketsDropped float, accuracy float)")
        con.commit()
    except:
        pass
    if(OPTION=="IPM[0-4].AckGen" or OPTION=="Switch_0_[0-2].IPM[134].AckGen"):
        OPTION="AckGenIn"
    if(OPTION=="OPM[0-4].AckGen" or OPTION=="Switch_0_[0-2].OPM[134].AckGen"):
        OPTION="AckGenOut"
    if(OPTION=="OPM[0-4].[^ToA][a-zA-Z][^t]" or OPTION=="Switch_0_[0-2].OPM[134].[^ToA][a-zA-Z][^t]"):
        OPTION="OPM General Logic"
    if(OPTION=="mutex" or OPTION=="Switch_0_[0-2].OPM[134].mutex"):
        OPTION="Arbiter"
    if(OPTION=="routeSelector" or OPTION=="Switch_0_[0-2].IPM[134].routeSelector"):
        OPTION="Packet Route Selector"
    if(OPTION=="reqGen" or OPTION=="Switch_0_[0-2].IPM[134].reqGen"):
        OPTION="Request Generator"
    if(OPTION=="Switch_0_[0-2].IPM[134].inputMousetrap"):
        OPTION="input Mousetrap"
    if(OPTION=="Switch_0_[0-2].OPM[134].outputMousetrap"):
        OPTION="output Mousetrap"
    if(OPTION=="TailDetect" or OPTION=="Switch_0_[0-2].OPM[134]TailDetect"):
        OPTION="Tail detector"
    cur.execute(f"insert into FaultInjResults values(\"{OPTION}\",{wrongmean},{accmean})")
    con.commit()
    con.close();


