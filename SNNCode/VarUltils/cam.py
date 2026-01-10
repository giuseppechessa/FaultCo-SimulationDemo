from dataclasses import dataclass
from typing import List, Dict
from collections import defaultdict
import sqlite3
import json
import os


@dataclass
class CAMEntry:
    source_neuron: int
    destination_neurons: List[int]



class SimpleCAMGenerator:
    """Generate CAM tables"""
    
    def __init__(self, neuron_lookup: Dict, all_connections: List):
        self.neuron_lookup = neuron_lookup
        self.all_connections = all_connections

    
    def generate_simple_cam_tables(self) -> Dict[int, Dict[int, List[int]]]:
        """
        Return: {dest_core: {source_global_idx: [dest_global_idx, ...]}}
        """
        cam_tables = defaultdict(lambda: defaultdict(list))
        for connection in self.all_connections:
            source_neuron_id = connection.source_neuron
            if source_neuron_id not in self.neuron_lookup:
                continue
                
            source_global = self.neuron_lookup[source_neuron_id].global_index
            source_core = connection.source_core
            
            for destination in connection.destinations:
                dest_core = destination.destination_core
                
                # filter out same core communication
                if dest_core == source_core:
                    continue
                
                for dest_local in destination.destination_neurons:
                    # get destination neuron ID
                    #if dest_core <= 3:  # lif1 layer
                    #    dest_neuron_id = f"lif1-{dest_local}"
                    #else:  # lif2 layer 
                    #    dest_neuron_id = f"lif2-{dest_local}"
                    ##
                    #if dest_neuron_id in self.neuron_lookup:
                    #    dest_global = self.neuron_lookup[dest_neuron_id].global_index
                    cam_tables[dest_core][source_global].append(dest_local)
        
        for core_id in cam_tables:
            for source_idx in cam_tables[core_id]:
                cam_tables[core_id][source_idx] = sorted(list(set(cam_tables[core_id][source_idx])))
        
        return dict(cam_tables)
    
    def generate_standard_cam_tables(self) -> Dict[int, List[CAMEntry]]:
        """
        return: {dest_core: [CAMEntry, ...]}
        """
        simple_tables = self.generate_simple_cam_tables()
        standard_tables = {}
        
        for core_id, core_data in simple_tables.items():
            standard_tables[core_id] = []
            for source_global, dest_globals in core_data.items():
                cam_entry = CAMEntry(
                    source_neuron=source_global,
                    destination_neurons=dest_globals
                )
                standard_tables[core_id].append(cam_entry)
            
            # sort entries by source neuron
            standard_tables[core_id].sort(key=lambda x: x.source_neuron)
        
        return standard_tables
    
    def display_cam_tables_with_filtering_stats(self, max_entries_per_core=25):
        """Generate and display CAM tables with filtering statistics"""

        total_connections = 0
        inter_core_connections = 0
        same_core_connections = 0
        
        for connection in self.all_connections:
            source_core = connection.source_core
            for destination in connection.destinations:
                dest_core = destination.destination_core
                connection_count = len(destination.destination_neurons)
                total_connections += connection_count
                
                if dest_core == source_core:
                    same_core_connections += connection_count
                else:
                    inter_core_connections += connection_count
        
        # Gen cam
        cam_tables = self.generate_standard_cam_tables()
        
        return cam_tables
    
    def save_cam_tables_to_database(self, cam_tables, db_path="./database/CAMTables.sql"):
        
        try:
            if os.path.exists(db_path):
                os.remove(db_path)
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE CAMTables(
                    Core INTEGER,
                    Source INTEGER,
                    Destinations TEXT
                )
            """)
            
            total_entries = 0
            for core_id in sorted(cam_tables.keys()):
                table = cam_tables[core_id]
                for entry in table:
                    
                    destinations_json = json.dumps(entry.destination_neurons)
                    
                    cursor.execute("""
                        INSERT INTO CAMTables VALUES(?, ?, ?)
                    """, (
                        core_id,
                        entry.source_neuron,
                        destinations_json
                    ))
                    total_entries += 1
            
            conn.commit()
            conn.close()
            
            print(f"Finished saving CAM tables")
            print(f"Path of database: {db_path}")
            print(f"Total entries: {total_entries}")
            print(f"Total cores: {len(cam_tables)}")
            
            return db_path
            
        except Exception as e:
            print(f"Error: {e}")
            return None
