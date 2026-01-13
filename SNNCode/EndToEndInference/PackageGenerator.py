import numpy as np
import time
import pandas as pd
import os
import sys
import torch
from collections import defaultdict, Counter
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
import sqlite3

from VarUltils.mapping import Mapping

# Package data structure for online generation
@dataclass
class Package:
    """
    Package containing 16-bit header and 16-bit payload
    Header: destination cores bitmap (16 bits)
    Payload: source neuron index (16 bits)
    """
    header: int  # 16-bit destination cores bitmap
    payload: int  # 16-bit source neuron index
    source_core: int  # which core generated this package
    time_step: int  # at which time step this package was generated

# PackageGenerator class - based on spk1, no hook mechanism
class PackageGenerator:

    def __init__(self, neuron_lookup=None, network_reference=None, max_cores=16, dataSx_path="./database/DataSx.sql", dataRx_path="./database/DataRx.sql",IdLookup=dict()):
        self.neuron_lookup = neuron_lookup
        self.network_reference = network_reference
        self.max_cores = max_cores
        self.current_time_step = 0
        self.IdLookup=IdLookup

        # Package storage
        self.packages_by_core = defaultdict(list)  # core_id -> List[Package]
        self.self_communication_connections = defaultdict(list)  # core_id -> List[Dict]

        # Database configuration
        self.dataSx_path = dataSx_path
        self.dataRx_path = dataRx_path
        self.dataSx_connection = None
        self.dataRx_connection = None
        self.dataSx_cursor = None
        self.dataRx_cursor = None

        # Initialize databases
        self.init_databases()

        # Precompute non-zero weight indices for recurrent and feedforward connections
        self.nonzero_recurrent = {}
        self.nonzero_feedforward = {}
        self._build_weight_indices()

    def getNonzeroMatrix(self):
        return self.nonzero_recurrent,self.nonzero_feedforward 


    def HelperRecurrFunc(self,recurrent_weights,StartIndex):
        recurrent_nonzero = torch.nonzero(recurrent_weights != 0)
        for dest, src in recurrent_nonzero:
            src_idx = src.item()+StartIndex
            if(self.neuron_lookup[self.IdLookup[src_idx]].source_core!=self.neuron_lookup[self.IdLookup[dest.item()+StartIndex]].source_core):
                if src_idx not in self.nonzero_recurrent:
                    self.nonzero_recurrent[src_idx] = []
                self.nonzero_recurrent[src_idx].append(dest.item()+StartIndex)
    def HelperFFFunc(self,feedforward_weights,StartIndex,StartIndex1):
        feedforward_nonzero = torch.nonzero(feedforward_weights != 0)
        for dest, src in feedforward_nonzero:
            src_idx = src.item()+StartIndex
            if(self.neuron_lookup[self.IdLookup[src_idx]].source_core!=self.neuron_lookup[self.IdLookup[dest.item()+StartIndex1]].source_core):
                if src_idx not in self.nonzero_feedforward:
                    self.nonzero_feedforward[src_idx] = []
                self.nonzero_feedforward[src_idx].append(dest.item() + StartIndex1)
            
    def _build_weight_indices(self):
        '''
        Precompute non-zero weight indices for recurrent and feedforward connections
        '''
        if self.network_reference is None:
            return

        try:
            # if has recurrent connections
            if hasattr(self.network_reference, 'lif1') and hasattr(self.network_reference.lif1, 'recurrent'):
                self.HelperRecurrFunc(self.network_reference.lif1.recurrent.weight.data,self.neuron_lookup["lif1-0"].global_index)
            if hasattr(self.network_reference, 'lif2') and hasattr(self.network_reference.lif2, 'recurrent'):
                self.HelperRecurrFunc(self.network_reference.lif2.recurrent.weight.data,self.neuron_lookup["lif2-0"].global_index)
            if hasattr(self.network_reference, 'lif3') and hasattr(self.network_reference.lif3, 'recurrent'):
                self.HelperRecurrFunc(self.network_reference.lif3.recurrent.weight.data,self.neuron_lookup["lif3-0"].global_index)

            # if has feedforward connections
            if hasattr(self.network_reference, 'fc2'):
                self.HelperFFFunc(self.network_reference.fc2.weight.data,self.neuron_lookup["lif1-0"].global_index,self.neuron_lookup["lif2-0"].global_index)
            if hasattr(self.network_reference, 'fc3'):
                self.HelperFFFunc(self.network_reference.fc3.weight.data,self.neuron_lookup["lif2-0"].global_index,self.neuron_lookup["lif3-0"].global_index)
            if hasattr(self.network_reference, 'fc4'):
                self.HelperFFFunc(self.network_reference.fc4.weight.data,self.neuron_lookup["lif3-0"].global_index,self.neuron_lookup["lif4-0"].global_index)
                
        except Exception as e:
            print("There has been an exception when building weight indices", e)
            self.nonzero_recurrent = {}
            self.nonzero_feedforward = {}

    def separate_self_communication(self, destination_cores, source_core):
        """Separate self-communication from inter-core communication"""
        inter_core_destinations = []
        has_self_communication = False

        for dest_core in destination_cores:
            if dest_core == source_core:
                has_self_communication = True
            else:
                inter_core_destinations.append(dest_core)

        return inter_core_destinations, has_self_communication

    def init_databases(self):
        """Initialize both DataSx and DataRx databases"""
        try:
            self.init_dataSx_database()
            self.init_dataRx_database()
            # print("PackageGenerator: Dual databases initialized successfully!")
        except Exception as e:
            print(f"PackageGenerator database initialization failed: {e}")
            raise

    def init_dataSx_database(self):
        """Initialize DataSx database for writing transmitted packages"""
        try:
            # Remove existing database file to start fresh
            if os.path.exists(self.dataSx_path):
                os.remove(self.dataSx_path)

            # Create new connection
            self.dataSx_connection = sqlite3.connect(self.dataSx_path)
            self.dataSx_cursor = self.dataSx_connection.cursor()

            # Create table
            self.dataSx_cursor.execute("""
                CREATE TABLE DataSx(
                    Core INTEGER, 
                    header INTEGER, 
                    payload INTEGER
                )
            """)
            self.dataSx_connection.commit()

        except Exception as e:
            print(f"DataSx database initialization failed: {e}")
            raise

    def init_dataRx_database(self):
        """Initialize DataRx database connection for reading received packages"""
        try:
            # Create connection (don't remove existing file as it's managed by external system)
            self.dataRx_connection = sqlite3.connect(self.dataRx_path)
            self.dataRx_cursor = self.dataRx_connection.cursor()

            # Create table if it doesn't exist
            self.dataRx_cursor.execute("""
                CREATE TABLE IF NOT EXISTS DataRx(
                    Core INTEGER, 
                    Source INTEGER, 
                    Destination INTEGER
                )
            """)
            self.dataRx_connection.commit()

        except Exception as e:
            print(f"DataRx database initialization failed: {e}")
            raise

    def close_databases(self):
        """Close both database connections safely"""
        try:
            if self.dataSx_connection:
                self.dataSx_connection.close()
                self.dataSx_connection = None
                self.dataSx_cursor = None

            if self.dataRx_connection:
                self.dataRx_connection.close()
                self.dataRx_connection = None
                self.dataRx_cursor = None

            # print("PackageGenerator: Database connections closed successfully!")
        except Exception as e:
            print(f"PackageGenerator error closing databases: {e}")

    def clear_dataSx_table(self):
        """Clear DataSx table for new time step"""
        try:
            if self.dataSx_cursor:
                self.dataSx_cursor.execute("DELETE FROM DataSx")
                self.dataSx_connection.commit()
        except Exception as e:
            print(f"Error clearing DataSx table: {e}")
    
    def insert_package_to_dataSx(self, packages: Package):
        """Insert package data into DataSx database"""
        try:
            if self.dataSx_cursor and len(packages)!=0:
                command=f"INSERT INTO DataSx VALUES ({packages[0].source_core}, {packages[0].header}, {packages[0].payload})"
                for package in packages[1:]:
                    command +=  f",({package.source_core}, {package.header}, {package.payload})"
                command += ";"
                self.dataSx_cursor.execute(command)
                self.dataSx_connection.commit()
        except Exception as e:
            print(f"Error inserting package to DataSx: {e}")

    def read_packages_from_dataRx(self) -> List[Dict]:
        """Read all packages from DataRx database"""
        try:
            conn = sqlite3.connect(self.dataRx_path)
            cursor = conn.cursor()
            cursor.execute("SELECT Core, Source, Destination FROM DataRx")
            rows = cursor.fetchall()
            conn.close()
            return [{"core": core, "source": source, "destination": destination} for core, source, destination in rows]
        except Exception as e:
            print(f"Error reading from DataRx: {e}")
            return []

    # Helper function to convert destination cores list to 16-bit bitmap
    def cores_to_bitmap(self, destination_cores: List[int]) -> int:
        """Convert list of destination cores to 16-bit bitmap"""
        bitmap = 0
        for core in destination_cores:
            if core < 16:  # Only support up to 16 cores
                bitmap |= (1 << core)
        return bitmap

    # Helper function to convert neuron index to 16-bit payload
    def neuron_to_payload(self, neuron_index: int) -> int:
        """Convert neuron index to 16-bit payload"""
        return neuron_index & 0xFFFF  # Mask to 16 bits

    def generate_packages(self, spk1: torch.Tensor, layerName):
        """
        Args:
            spk1: Tensor of shape (batch_size, num_neurons), the batch size is 1, otherwise the rest data would be ignored

            Input: Spike tensor from lif1 layer
        """
        lenComm=0;
        inter_package = []
        if spk1.dim() == 3:
            spk1 = spk1[0][0]  # Only take first batch element, assuming batch size is 1
        elif spk1.dim() == 2:
            spk1 = spk1[0]  # Only take first batch element, assuming batch size is 1

        # Find all spiking neurons
        spiking_neurons = torch.nonzero(spk1, as_tuple=False).flatten()
        # Generate packages for each spiking neuron
        for neuron_idx in spiking_neurons:
            neuron_idx = neuron_idx.item()
            layer_name = layerName 
            neuron_id = f"{layer_name}-{neuron_idx}"

            if neuron_id in self.neuron_lookup:
                lookup = self.neuron_lookup[neuron_id]

                # Only generate package if there are destination cores
                if lookup.destination_cores:
                    # Separate self-communication from inter-core communication
                    inter_core_destinations,_  = self.separate_self_communication(
                        lookup.destination_cores, lookup.source_core
                    )

                    # Handle inter-core communication (needs network transmission)
                    if inter_core_destinations:
                        inter_package.append(Package(
                            header=self.cores_to_bitmap(inter_core_destinations),
                            payload=self.neuron_to_payload(lookup.global_index),
                            source_core=lookup.source_core,
                            time_step=self.current_time_step
                        ))
                        lenComm+=1;

        self.insert_package_to_dataSx(inter_package)
        return lenComm

    def get_current_transmitted_packages(self) -> List[Dict]:
        """Get current transmitted packages from DataSx"""
        try:
            if self.dataSx_cursor:
                self.dataSx_cursor.execute("SELECT Core, header, payload FROM DataSx")
                rows = self.dataSx_cursor.fetchall()

                packages = []
                for row in rows:
                    packages.append({
                        'core': row[0],
                        'header': row[1], 
                        'payload': row[2]
                    })
                return packages
            return []
        except Exception as e:
            print(f"Error getting transmitted packages: {e}")
            return []

    def get_current_self_communication_connections(self) -> Dict[int, List[Dict]]:
        """Get self-communication connections generated in current time step"""
        current_self_connections = {}
        for core_id, connections in self.self_communication_connections.items():
            current_self_connections[core_id] = [c for c in connections if c['time_step'] == self.current_time_step]
        return current_self_connections

    def step(self, keep_database=False):
        """
        Args:
            keep_database: If False, clear DataSx table for new time step (default behavior)
                           If True, keep existing data in DataSx for accumulation
        """
        self.current_time_step += 1
        # Clear DataSx table only if not keeping database
        if not keep_database:
            self.packages_by_core.clear()
            self.clear_dataSx_table()

    def reset(self):
        """Reset the package generator for a new inference"""
        self.current_time_step = 0
        self.packages_by_core.clear()
        self.self_communication_connections.clear()

        # # Reset DataSx for new inference
        # self.clear_dataSx_table()
        self.close_databases()

    def __del__(self):
        """Destructor to ensure databases are closed"""
        self.close_databases()



class HardwareStateCorrector:
    def __init__(self, network, neuron_lookup, all_connections,IdLookup,nonzero_recurrent, nonzero_feedforward,package_decoder=None):
        """
        Args:
            network: SNN network
            neuron_lookup: neuron lookup table
            all_connections: complete neuron connections info
            package_decoder: package decoder instance, this item is useless, can be deleted
        """
        self.network = network
        self.neuron_lookup = neuron_lookup
        self.all_connections = all_connections
        self.package_decoder = package_decoder
        self.IdLookup=IdLookup
        self.nonzero_recurrent = nonzero_recurrent
        self.nonzero_feedforward = nonzero_feedforward

    def build_expected_connections(self, predicted_spikes):
        """
        Args:
            predicted_spikes: Dict[str, torch.Tensor]
        Returns:
            Dict[int, Counter] - {dest_global_idx: Counter({source1: count1, source2: count2, ...})}
        """
        # Use Counter to process the duplicate connections
        expected = defaultdict(Counter)
        LIFS = ["lif1","lif2","lif3"]
        for lif in LIFS:
            if lif in predicted_spikes:
                spikes = predicted_spikes[lif].squeeze()  # [100]
                spiking_sources = torch.nonzero(spikes > 0, as_tuple=False).flatten() # [n1, n2, ...] local index in the lif1 layer
                
                # process lif1 recurrent connections (lif1 -> lif1)
                for src_local in spiking_sources:
                    src_global =  self.neuron_lookup[f"{lif}-{src_local.item()}"].global_index
                    if src_global in self.nonzero_recurrent:
                        for dest_global in self.nonzero_recurrent[src_global]:
                            
                            # expected[dest_global].add(src_global)
                            expected[dest_global][src_global] += 1

                    if src_global in self.nonzero_feedforward:
                        for dest_global in self.nonzero_feedforward[src_global]:
                            # expected[dest_global].add(src_global)
                            expected[dest_global][src_global] += 1

        return dict(expected)

    def build_actual_connections_from_packages(self, hardware_packages):
        """
        Build actual connections from hardware packages
        Args:
            hardware_packages: List[Dict] - [{'source': int, 'destination': int}, ...]
        Returns:
            Dict[int, Counter] - {dest_global_idx: Counter({source1: count1, source2: count2, ...})}
        """

        # Use Counter to process the duplicate connections
        actual = defaultdict(Counter)

        for pkg in hardware_packages:
            source_global = pkg['source']
            dest_global = pkg['destination']
            # actual[dest_global].add(source_global)
            actual[dest_global][source_global] += 1

        return dict(actual)


    def generate_corrections_from_packages(self, predicted_spikes, hardware_packages, self_communication_connections=None):
        """
        Args:
            predicted_spikes: Dict[str, torch.Tensor] - predicted spikes
            hardware_packages: List[Dict] - hardware packages
            self_communication_connections: Dict[int, List[Dict]] - self-communication connections,  {'source': global_idx, 'destination': dest_global_idx, 'time_step': t}
        Returns:
            Tuple[List, List] - (cur_sub, cur_add) corrections should be applied
        """

        expected_connections = self.build_expected_connections(predicted_spikes)
        actual_connections = self.build_actual_connections_from_packages(hardware_packages)
        cur_sub = []  # Remove redundant connection contributions
        cur_add = []  # Fill in missing connection contributions
        if(actual_connections==expected_connections):
            return cur_sub,cur_add

        # Obtain all destination neurons involved
        all_destinations = set(expected_connections.keys()) | set(actual_connections.keys())
        for dest_global in all_destinations:
            expected_count = expected_connections.get(dest_global, Counter())
            actual_count = actual_connections.get(dest_global, Counter())
            all_sources = set(expected_count.keys()) | set(actual_count.keys())

            # For each source neuron, compare expected and actual counts
            for source_global in all_sources:
                expected_cnt = expected_count[source_global]
                actual_cnt = actual_count[source_global]
                diff = actual_cnt - expected_cnt
                if diff > 0:
                    for _ in range(diff):
                        cur_add.append([1.0, source_global, dest_global])
                elif diff < 0:
                    for _ in range(-diff):
                        cur_sub.append([1.0, source_global, dest_global])

        return cur_sub, cur_add
