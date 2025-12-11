import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

class SupplyChainEnvironment:
    """Custom environment for supply chain optimization"""
    def __init__(self, supply_chain_graph):
        self.supply_chain = supply_chain_graph
        self.state_size = self._get_state_size()
        self.action_size = self._get_action_size()
        self.current_step = 0
        self.max_steps = 100
        
    def _get_state_size(self):
        """Get state representation size"""
        # State includes: node capacities, costs, disruption status, etc.
        num_nodes = len(self.supply_chain.graph.nodes())
        num_edges = len(self.supply_chain.graph.edges())
        return num_nodes * 5 + num_edges * 4
    
    def _get_action_size(self):
        """Get action space size"""
        # Actions: reroute shipments, adjust inventory, activate backup suppliers
        return len(self.supply_chain.graph.nodes()) * 3
    
    def reset(self):
        """Reset environment to initial state"""
        self.current_step = 0
        return self._get_state()
    
    def step(self, action):
        """Take action in environment"""
        self._apply_action(action)
        
        # Simulate one time step
        self._simulate_time_step()
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if done
        done = self.current_step >= self.max_steps
        
        # Get next state
        next_state = self._get_state()
        
        self.current_step += 1
        
        return next_state, reward, done, {}
    
    def _get_state(self):
        """Get current state representation"""
        state = []
        
        # Node features
        for node_id in self.supply_chain.graph.nodes():
            node_state = [
                self.supply_chain.node_capacities[node_id],
                self.supply_chain.node_costs[node_id],
                float(self.supply_chain.disruption_status[node_id]['disrupted']),
                self.supply_chain.disruption_status[node_id]['reliability'],
                self.supply_chain.disruption_status[node_id].get('recovery_time', 0)
            ]
            state.extend(node_state)
        
        # Edge features
        for u, v, data in self.supply_chain.graph.edges(data=True):
            edge_state = [
                data.get('capacity', 0),
                data.get('lead_time', 0),
                data.get('cost', 0),
                float(data.get('disrupted', False))
            ]
            state.extend(edge_state)
            
        return np.array(state, dtype=np.float32)
    
    def _apply_action(self, action):
        """Apply RL agent action to supply chain"""
        # This is a simplified action space
        # In practice, you'd have more sophisticated actions
        action_idx = np.argmax(action)
        num_nodes = len(self.supply_chain.graph.nodes())
        
        if action_idx < num_nodes:
            # Activate backup capacity for a node
            node_id = list(self.supply_chain.graph.nodes())[action_idx]
            self.supply_chain.node_capacities[node_id] *= 1.1
        elif action_idx < 2 * num_nodes:
            # Reroute shipments
            pass  # Implement rerouting logic
        else:
            # Adjust inventory levels
            pass  # Implement inventory adjustment logic
    
    def _simulate_time_step(self):
        """Simulate one time step of supply chain operations"""
        # Reduce recovery times
        for node_id, status in self.supply_chain.disruption_status.items():
            if status['disrupted'] and status['recovery_time'] > 0:
                status['recovery_time'] -= 1
                if status['recovery_time'] == 0:
                    status['disrupted'] = False
                    # Restore capacity
                    self.supply_chain.node_capacities[node_id] *= 1.0/(1 - status.get('severity', 0))
        
        # Reduce edge recovery times
        for u, v, data in self.supply_chain.graph.edges(data=True):
            if data.get('disrupted', False) and 'recovery_duration' in data:
                data['recovery_duration'] -= 1
                if data['recovery_duration'] == 0:
                    data['disrupted'] = False
                    data['capacity'] *= 1.0/(1 - data.get('severity', 0))
    
    def _calculate_reward(self):
        """Calculate reward based on supply chain performance"""
        metrics = self.supply_chain.calculate_metrics()
        
        # Reward based on resilience and cost
        reward = (
            metrics['resilience_score'] * 100 +
            metrics['service_level'] * 50 -
            metrics['total_cost'] * 0.01
        )
        
        # Penalty for disruptions
        disrupted_nodes = sum(1 for status in self.supply_chain.disruption_status.values() 
                            if status['disrupted'])
        reward -= disrupted_nodes * 10
        
        return reward

class DQNAgent(nn.Module):
    """Deep Q-Network agent for supply chain optimization"""
    def __init__(self, state_size, action_size):
        super(DQNAgent, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        
        self.network = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
        
    def forward(self, state):
        return self.network(state)

class RLAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQNAgent(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.randn(self.action_size)
        
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action_values = self.model(state)
        return action_values.cpu().numpy()
    
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in batch:
            state = torch.FloatTensor(state).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            reward = torch.FloatTensor([reward]).to(self.device)
            
            target = reward
            if not done:
                next_q_value = self.model(next_state).max()
                target = reward + self.gamma * next_q_value
            
            current_q_value = self.model(state)
            loss = self.criterion(current_q_value.mean(), target)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def train(self, env, episodes=100):
        for episode in range(episodes):
            state = env.reset()
            total_reward = 0
            
            for step in range(env.max_steps):
                action = self.act(state)
                next_state, reward, done, _ = env.step(action)
                
                self.remember(state, action, reward, next_state, done)
                self.replay()
                
                state = next_state
                total_reward += reward
                
                if done:
                    break
            
            print(f"Episode {episode + 1}/{episodes}, Reward: {total_reward:.2f}, Epsilon: {self.epsilon:.3f}")