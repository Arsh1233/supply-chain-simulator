import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import random
from scipy.optimize import linprog

class SupplyChainOptimizer:
    def __init__(self, supply_chain_graph):
        self.supply_chain = supply_chain_graph
        self.best_solutions = []
        
    def optimize_flows(self, demand: Dict[str, float]) -> Dict[Tuple[str, str], float]:
        """Optimize material flows using linear programming"""
        
        # Extract nodes and edges
        nodes = list(self.supply_chain.graph.nodes())
        edges = list(self.supply_chain.graph.edges())
        
        # Create cost vector
        c = []
        for u, v in edges:
            cost = self.supply_chain.graph.edges[u, v]['cost']
            if self.supply_chain.graph.edges[u, v].get('disrupted', False):
                cost *= 2  # Penalty for disrupted edges
            c.append(cost)
        
        # Create capacity constraints
        A_ub = []
        b_ub = []
        
        # Edge capacity constraints
        for i, (u, v) in enumerate(edges):
            constraint = [0] * len(edges)
            constraint[i] = 1
            A_ub.append(constraint)
            capacity = self.supply_chain.graph.edges[u, v]['capacity']
            if self.supply_chain.graph.edges[u, v].get('disrupted', False):
                capacity *= (1 - self.supply_chain.graph.edges[u, v].get('severity', 0))
            b_ub.append(capacity)
        
        # Node capacity constraints
        for node in nodes:
            # Sum of outgoing flows from node
            constraint = [0] * len(edges)
            for i, (u, v) in enumerate(edges):
                if u == node:
                    constraint[i] = 1
            if any(constraint):  # Only add if node has outgoing edges
                A_ub.append(constraint)
                capacity = self.supply_chain.node_capacities[node]
                if self.supply_chain.disruption_status[node]['disrupted']:
                    capacity *= (1 - self.supply_chain.disruption_status[node].get('severity', 0))
                b_ub.append(capacity)
        
        # Demand satisfaction constraints (equality)
        A_eq = []
        b_eq = []
        
        # For each retailer, ensure demand is met
        retailers = [n for n, t in self.supply_chain.node_types.items() 
                    if t.value == 'retailer']
        
        for retailer in retailers:
            if retailer in demand:
                constraint = [0] * len(edges)
                # Find edges ending at retailer
                for i, (u, v) in enumerate(edges):
                    if v == retailer:
                        constraint[i] = 1
                A_eq.append(constraint)
                b_eq.append(demand[retailer])
        
        # Solve linear program
        bounds = [(0, None) for _ in range(len(edges))]
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, 
                        bounds=bounds, method='highs')
        
        if result.success:
            flows = {}
            for i, (u, v) in enumerate(edges):
                flows[(u, v)] = result.x[i]
            return flows
        else:
            print("Optimization failed:", result.message)
            return {}
    
    def find_alternative_routes(self, source: str, target: str, 
                               current_route: List[str]) -> List[List[str]]:
        """Find alternative routes between source and target"""
        all_routes = []
        
        # Find all simple paths
        try:
            import networkx as nx
            paths = nx.all_simple_paths(self.supply_chain.graph, source, target, cutoff=len(current_route)+2)
            all_routes = list(paths)
        except:
            pass
        
        # Filter out disrupted routes
        viable_routes = []
        for route in all_routes:
            if route != current_route:
                viable = True
                for i in range(len(route)-1):
                    edge_data = self.supply_chain.graph.edges[route[i], route[i+1]]
                    if edge_data.get('disrupted', False):
                        viable = False
                        break
                if viable:
                    viable_routes.append(route)
        
        return viable_routes
    
    def calculate_robustness_metrics(self) -> Dict:
        """Calculate robustness metrics for the supply chain"""
        metrics = {}
        
        # Network density
        num_nodes = len(self.supply_chain.graph.nodes())
        num_edges = len(self.supply_chain.graph.edges())
        max_edges = num_nodes * (num_nodes - 1)
        metrics['network_density'] = num_edges / max_edges if max_edges > 0 else 0
        
        # Redundancy score
        redundancy = 0
        for node in self.supply_chain.graph.nodes():
            in_degree = self.supply_chain.graph.in_degree(node)
            out_degree = self.supply_chain.graph.out_degree(node)
            redundancy += (in_degree + out_degree) / (2 * (num_nodes - 1))
        metrics['redundancy_score'] = redundancy / num_nodes if num_nodes > 0 else 0
        
        # Critical node identification
        critical_nodes = []
        for node in self.supply_chain.graph.nodes():
            # Calculate betweenness centrality (simplified)
            try:
                import networkx as nx
                centrality = nx.betweenness_centrality(self.supply_chain.graph)[node]
                if centrality > 0.1:  # Threshold for criticality
                    critical_nodes.append({
                        'node': node,
                        'centrality': centrality,
                        'type': self.supply_chain.node_types[node].value
                    })
            except:
                pass
        
        metrics['critical_nodes'] = sorted(critical_nodes, 
                                          key=lambda x: x['centrality'], 
                                          reverse=True)[:5]
        
        return metrics
    
    def optimize_inventory(self, demand_forecast: Dict[str, List[float]], 
                          service_level: float = 0.95) -> Dict[str, float]:
        """Optimize inventory levels using newsvendor model"""
        optimal_inventory = {}
        
        for node, demands in demand_forecast.items():
            if node not in self.supply_chain.node_types:
                continue
                
            # Convert to numpy array
            demands_array = np.array(demands)
            
            # Calculate optimal inventory level
            mean_demand = np.mean(demands_array)
            std_demand = np.std(demands_array)
            
            # For service level probability
            from scipy.stats import norm
            z_score = norm.ppf(service_level)
            
            # Safety stock calculation
            lead_time = 7  # Assume 7 days lead time
            safety_stock = z_score * std_demand * np.sqrt(lead_time)
            
            optimal_inventory[node] = mean_demand * lead_time + safety_stock
        
        return optimal_inventory