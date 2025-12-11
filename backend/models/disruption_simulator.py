import numpy as np
import random
from enum import Enum
from typing import List, Dict, Tuple
import pandas as pd

class DisruptionType(Enum):
    NATURAL_DISASTER = "natural_disaster"
    PANDEMIC = "pandemic"
    GEOPOLITICAL = "geopolitical"
    TRANSPORTATION = "transportation"
    CYBER_ATTACK = "cyber_attack"
    SUPPLIER_BANKRUPTCY = "supplier_bankruptcy"

class DisruptionSimulator:
    def __init__(self):
        self.disruption_profiles = self._create_disruption_profiles()
        self.historical_patterns = self._load_historical_patterns()
        
    def _create_disruption_profiles(self) -> Dict:
        """Create profiles for different disruption types"""
        return {
            DisruptionType.NATURAL_DISASTER: {
                'severity_range': (0.3, 0.9),
                'duration_range': (7, 90),
                'recovery_pattern': 'exponential',
                'geographic_spread': 'regional',
                'propagation_speed': 'fast'
            },
            DisruptionType.PANDEMIC: {
                'severity_range': (0.4, 0.8),
                'duration_range': (30, 365),
                'recovery_pattern': 'sigmoid',
                'geographic_spread': 'global',
                'propagation_speed': 'medium'
            },
            DisruptionType.GEOPOLITICAL: {
                'severity_range': (0.5, 1.0),
                'duration_range': (30, 180),
                'recovery_pattern': 'step',
                'geographic_spread': 'country',
                'propagation_speed': 'instant'
            },
            DisruptionType.TRANSPORTATION: {
                'severity_range': (0.2, 0.7),
                'duration_range': (3, 30),
                'recovery_pattern': 'linear',
                'geographic_spread': 'local',
                'propagation_speed': 'instant'
            }
        }
    
    def _load_historical_patterns(self) -> pd.DataFrame:
        """Load historical disruption patterns"""
        # In practice, this would load from a database
        patterns = {
            'event': ['COVID-19', 'Suez Canal Blockage', 'Ukraine Conflict', 'Japan Earthquake'],
            'type': ['pandemic', 'transportation', 'geopolitical', 'natural_disaster'],
            'severity': [0.7, 0.5, 0.9, 0.8],
            'duration_days': [365, 7, 180, 30],
            'economic_impact_billion': [9000, 10, 150, 200]
        }
        return pd.DataFrame(patterns)
    
    def generate_disruption(self, disruption_type: DisruptionType, 
                           location: Tuple[float, float] = None,
                           severity: float = None) -> Dict:
        """Generate a disruption event"""
        profile = self.disruption_profiles[disruption_type]
        
        if severity is None:
            severity = random.uniform(*profile['severity_range'])
        
        duration = random.randint(*profile['duration_range'])
        
        return {
            'type': disruption_type,
            'severity': severity,
            'duration': duration,
            'location': location,
            'recovery_pattern': profile['recovery_pattern'],
            'propagation_speed': profile['propagation_speed'],
            'geographic_spread': profile['geographic_spread']
        }
    
    def simulate_cascading_effects(self, initial_disruption: Dict, 
                                  supply_chain_graph) -> List[Dict]:
        """Simulate cascading effects through the supply chain"""
        disruptions = [initial_disruption]
        
        # Simulate propagation based on network connectivity
        affected_nodes = set()
        
        # Start from initial location if specified
        if initial_disruption['location']:
            # Find nearest nodes to disruption location
            for node_id, loc in supply_chain_graph.node_locations.items():
                distance = self._calculate_distance(initial_disruption['location'], loc)
                if distance < 100:  # Within 100km
                    affected_nodes.add(node_id)
        
        # Propagate through network
        for node_id in affected_nodes:
            # Reduce severity with distance
            cascade_severity = initial_disruption['severity'] * 0.7
            
            disruption = {
                'type': initial_disruption['type'],
                'severity': cascade_severity,
                'duration': initial_disruption['duration'] * 0.5,
                'node_id': node_id,
                'is_cascading': True
            }
            disruptions.append(disruption)
            
            # Propagate to connected nodes
            for neighbor in supply_chain_graph.graph.neighbors(node_id):
                if neighbor not in affected_nodes:
                    affected_nodes.add(neighbor)
        
        return disruptions
    
    def _calculate_distance(self, loc1: Tuple[float, float], 
                          loc2: Tuple[float, float]) -> float:
        """Calculate Euclidean distance between two locations"""
        return np.sqrt((loc1[0] - loc2[0])**2 + (loc1[1] - loc2[1])**2)
    
    def generate_scenario(self, scenario_type: str = "moderate") -> List[Dict]:
        """Generate a complete disruption scenario"""
        scenarios = {
            "mild": {
                "num_disruptions": 2,
                "max_severity": 0.4,
                "types": [DisruptionType.TRANSPORTATION, DisruptionType.SUPPLIER_BANKRUPTCY]
            },
            "moderate": {
                "num_disruptions": 4,
                "max_severity": 0.7,
                "types": [DisruptionType.NATURAL_DISASTER, DisruptionType.GEOPOLITICAL,
                         DisruptionType.TRANSPORTATION, DisruptionType.CYBER_ATTACK]
            },
            "severe": {
                "num_disruptions": 6,
                "max_severity": 0.9,
                "types": [DisruptionType.PANDEMIC, DisruptionType.GEOPOLITICAL,
                         DisruptionType.NATURAL_DISASTER, DisruptionType.CYBER_ATTACK,
                         DisruptionType.TRANSPORTATION, DisruptionType.SUPPLIER_BANKRUPTCY]
            }
        }
        
        scenario_config = scenarios[scenario_type]
        disruptions = []
        
        for i in range(scenario_config["num_disruptions"]):
            disruption_type = random.choice(scenario_config["types"])
            severity = random.uniform(0.2, scenario_config["max_severity"])
            
            disruption = self.generate_disruption(
                disruption_type=disruption_type,
                severity=severity
            )
            disruptions.append(disruption)
        
        return disruptions