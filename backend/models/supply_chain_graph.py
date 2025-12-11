import numpy as np
import networkx as nx
import pandas as pd
from typing import Dict, List, Tuple
import random
from enum import Enum

class NodeType(Enum):
    SUPPLIER = "supplier"
    MANUFACTURER = "manufacturer"
    DISTRIBUTION_CENTER = "distribution_center"
    RETAILER = "retailer"
    CUSTOMER = "customer"

class SupplyChainGraph:
    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_types = {}
        self.node_capacities = {}
        self.node_costs = {}
        self.node_locations = {}
        self.disruption_status = {}
        
    def add_node(self, node_id: str, node_type: NodeType, 
                 capacity: float, cost: float, location: Tuple[float, float],
                 reliability: float = 0.95):
        """Add a node to the supply chain graph"""
        self.graph.add_node(node_id)
        self.node_types[node_id] = node_type
        self.node_capacities[node_id] = capacity
        self.node_costs[node_id] = cost
        self.node_locations[node_id] = location
        self.disruption_status[node_id] = {
            'disrupted': False,
            'reliability': reliability,
            'recovery_time': 0
        }
        
    def add_edge(self, source: str, target: str, 
                 capacity: float, lead_time: int, 
                 cost_per_unit: float, reliability: float = 0.9):
        """Add a transportation edge between nodes"""
        self.graph.add_edge(source, target)
        self.graph.edges[source, target]['capacity'] = capacity
        self.graph.edges[source, target]['lead_time'] = lead_time
        self.graph.edges[source, target]['cost'] = cost_per_unit
        self.graph.edges[source, target]['reliability'] = reliability
        self.graph.edges[source, target]['disrupted'] = False
        
    def apply_disruption(self, node_id: str, severity: float, duration: int):
        """Apply disruption to a node"""
        if node_id in self.disruption_status:
            self.disruption_status[node_id]['disrupted'] = True
            self.disruption_status[node_id]['severity'] = severity
            self.disruption_status[node_id]['recovery_time'] = duration
            # Reduce capacity based on severity
            self.node_capacities[node_id] *= (1 - severity)
            
    def apply_edge_disruption(self, source: str, target: str, 
                             severity: float, duration: int):
        """Apply disruption to an edge"""
        if self.graph.has_edge(source, target):
            self.graph.edges[source, target]['disrupted'] = True
            self.graph.edges[source, target]['severity'] = severity
            self.graph.edges[source, target]['recovery_duration'] = duration
            self.graph.edges[source, target]['capacity'] *= (1 - severity)
            
    def calculate_metrics(self) -> Dict:
        """Calculate key supply chain metrics"""
        # Calculate resilience score
        resilience_score = self._calculate_resilience()
        
        # Calculate cost metrics
        total_cost = sum(self.node_costs.values())
        
        # Calculate service level
        service_level = self._calculate_service_level()
        
        return {
            'resilience_score': resilience_score,
            'total_cost': total_cost,
            'service_level': service_level,
            'node_count': len(self.graph.nodes()),
            'edge_count': len(self.graph.edges())
        }
    
    def _calculate_resilience(self) -> float:
        """Calculate overall resilience score"""
        node_resilience = []
        for node_id, status in self.disruption_status.items():
            if status['disrupted']:
                resilience = status['reliability'] * (1 - status.get('severity', 0))
            else:
                resilience = status['reliability']
            node_resilience.append(resilience)
            
        edge_resilience = []
        for u, v, data in self.graph.edges(data=True):
            if data.get('disrupted', False):
                resilience = data['reliability'] * (1 - data.get('severity', 0))
            else:
                resilience = data['reliability']
            edge_resilience.append(resilience)
            
        all_resilience = node_resilience + edge_resilience
        return np.mean(all_resilience) if all_resilience else 1.0
    
    def _calculate_service_level(self) -> float:
        """Calculate service level based on disruptions"""
        operational_nodes = sum(1 for status in self.disruption_status.values() 
                              if not status['disrupted'])
        total_nodes = len(self.disruption_status)
        return operational_nodes / total_nodes if total_nodes > 0 else 1.0
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert graph to pandas dataframe for visualization"""
        nodes_data = []
        for node in self.graph.nodes():
            nodes_data.append({
                'id': node,
                'type': self.node_types[node].value,
                'capacity': self.node_capacities[node],
                'cost': self.node_costs[node],
                'disrupted': self.disruption_status[node]['disrupted'],
                'x': self.node_locations[node][0],
                'y': self.node_locations[node][1]
            })
            
        edges_data = []
        for u, v, data in self.graph.edges(data=True):
            edges_data.append({
                'source': u,
                'target': v,
                'capacity': data.get('capacity', 0),
                'lead_time': data.get('lead_time', 0),
                'cost': data.get('cost', 0),
                'disrupted': data.get('disrupted', False)
            })
            
        return pd.DataFrame(nodes_data), pd.DataFrame(edges_data)