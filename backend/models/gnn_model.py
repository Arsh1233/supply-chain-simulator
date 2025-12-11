import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
import numpy as np

class SupplyChainGNN(nn.Module):
    """Graph Neural Network for supply chain optimization"""
    def __init__(self, num_node_features=5, num_edge_features=4, hidden_dim=64, output_dim=1):
        super(SupplyChainGNN, self).__init__()
        
        # Node feature processing
        self.node_encoder = nn.Sequential(
            nn.Linear(num_node_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # GCN layers
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        
        # Edge feature processing
        self.edge_encoder = nn.Sequential(
            nn.Linear(num_edge_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, data):
        # Process node features
        x = self.node_encoder(data.x)
        
        # Apply GCN layers
        x = F.relu(self.conv1(x, data.edge_index))
        x = self.conv2(x, data.edge_index)
        
        # Pool node features
        node_pool = global_mean_pool(x, data.batch)
        
        # Process edge features if available
        if hasattr(data, 'edge_attr'):
            edge_features = self.edge_encoder(data.edge_attr)
            edge_pool = global_mean_pool(edge_features, data.edge_index[0])
            # Combine node and edge features
            combined = torch.cat([node_pool, edge_pool], dim=1)
        else:
            combined = node_pool
        
        # Make prediction
        output = self.prediction_head(combined)
        
        return output

class MultiAgentGNN:
    """Multi-agent system using GNN for supply chain coordination"""
    def __init__(self, num_agents=5):
        self.num_agents = num_agents
        self.agents = []
        
        for _ in range(num_agents):
            agent = SupplyChainGNN()
            self.agents.append(agent)
            
        self.coordination_network = nn.Sequential(
            nn.Linear(num_agents * 64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_agents)
        )
    
    def coordinate_agents(self, agent_outputs):
        """Coordinate multiple agents' decisions"""
        combined = torch.cat(agent_outputs, dim=1)
        coordination_scores = self.coordination_network(combined)
        return coordination_scores
    
    def predict_disruption_risk(self, supply_chain_data):
        """Predict disruption risk for each node"""
        risks = []
        for agent in self.agents:
            risk = agent(supply_chain_data)
            risks.append(risk)
        
        # Combine risks with coordination
        coordinated_risks = self.coordinate_agents(torch.stack(risks))
        return coordinated_risks