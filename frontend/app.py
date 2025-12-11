import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.data.sample_data import create_sample_supply_chain, create_disruption_scenario
from backend.models.disruption_simulator import DisruptionSimulator, DisruptionType
from backend.core.optimizer import SupplyChainOptimizer
import time

# Page configuration
st.set_page_config(
    page_title="Resilient Supply Chain Simulator",
    page_icon="🔄",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #3B82F6;
    }
    .stButton>button {
        width: 100%;
        background-color: #3B82F6;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'supply_chain' not in st.session_state:
    st.session_state.supply_chain = create_sample_supply_chain()
if 'disruptions' not in st.session_state:
    st.session_state.disruptions = []
if 'optimizer' not in st.session_state:
    st.session_state.optimizer = SupplyChainOptimizer(st.session_state.supply_chain)
if 'simulator' not in st.session_state:
    st.session_state.simulator = DisruptionSimulator()

def main():
    st.markdown("<h1 class='main-header'>🌐 Resilient Supply Chain Simulator with AI Agents</h1>", 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Supply Chain Selection
        st.subheader("Supply Chain Type")
        sc_type = st.selectbox(
            "Select Supply Chain Type",
            ["Electronics", "Automotive", "Pharmaceutical", "Retail", "Custom"]
        )
        
        # Disruption Scenario
        st.subheader("Disruption Scenario")
        scenario_type = st.select_slider(
            "Scenario Severity",
            options=["mild", "moderate", "severe"],
            value="moderate"
        )
        
        if st.button("🚨 Apply Disruption Scenario"):
            with st.spinner("Applying disruptions..."):
                create_disruption_scenario(st.session_state.supply_chain)
                st.session_state.disruptions = st.session_state.simulator.generate_scenario(scenario_type)
                st.success(f"Applied {len(st.session_state.disruptions)} disruptions!")
        
        # AI Optimization
        st.subheader("AI Optimization")
        if st.button("🤖 Run AI Optimization"):
            with st.spinner("Running AI optimization..."):
                # Simulate optimization
                time.sleep(2)
                st.success("Optimization complete!")
        
        # Reset Button
        if st.button("🔄 Reset Simulation"):
            st.session_state.supply_chain = create_sample_supply_chain()
            st.session_state.disruptions = []
            st.success("Simulation reset!")
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🌍 Network View", "📈 Analytics", "🤖 AI Insights"])
    
    with tab1:
        display_dashboard()
    
    with tab2:
        display_network_view()
    
    with tab3:
        display_analytics()
    
    with tab4:
        display_ai_insights()

def display_dashboard():
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate metrics
    metrics = st.session_state.supply_chain.calculate_metrics()
    
    with col1:
        st.metric(
            label="Resilience Score",
            value=f"{metrics['resilience_score']:.2%}",
            delta="+5%" if metrics['resilience_score'] > 0.8 else "-3%"
        )
    
    with col2:
        st.metric(
            label="Service Level",
            value=f"{metrics['service_level']:.2%}",
            delta="+2%" if metrics['service_level'] > 0.9 else "-4%"
        )
    
    with col3:
        st.metric(
            label="Total Cost",
            value=f"${metrics['total_cost']:,.0f}",
            delta="-5%" if metrics['total_cost'] < 100000 else "+8%"
        )
    
    with col4:
        disrupted = sum(1 for status in st.session_state.supply_chain.disruption_status.values() 
                       if status['disrupted'])
        st.metric(
            label="Active Disruptions",
            value=disrupted,
            delta=f"+{disrupted}" if disrupted > 0 else "0"
        )
    
    # Charts row
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Performance Over Time")
        
        # Generate sample time series data
        time_data = pd.DataFrame({
            'Day': range(1, 31),
            'Resilience': np.random.normal(0.85, 0.05, 30).cumsum() / np.arange(1, 31),
            'Cost Efficiency': np.random.normal(0.9, 0.03, 30).cumsum() / np.arange(1, 31),
            'Service Level': np.random.normal(0.95, 0.02, 30).cumsum() / np.arange(1, 31)
        })
        
        fig = px.line(time_data, x='Day', y=['Resilience', 'Cost Efficiency', 'Service Level'],
                     title="Key Metrics Trend",
                     labels={'value': 'Score', 'variable': 'Metric'},
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("⚠️ Disruption Impact")
        
        if st.session_state.disruptions:
            disruption_df = pd.DataFrame(st.session_state.disruptions)
            disruption_df['severity'] = disruption_df['severity'].apply(lambda x: x * 100)
            
            fig = px.bar(disruption_df, x='type', y='severity',
                        color='type',
                        title="Disruption Severity by Type",
                        labels={'severity': 'Severity (%)', 'type': 'Disruption Type'})
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No active disruptions. Apply a scenario from the sidebar.")
    
    # Current disruptions
    st.markdown("---")
    st.subheader("🔴 Current Disruptions")
    
    disrupted_nodes = []
    for node_id, status in st.session_state.supply_chain.disruption_status.items():
        if status['disrupted']:
            disrupted_nodes.append({
                'Node': node_id,
                'Type': st.session_state.supply_chain.node_types[node_id].value,
                'Severity': f"{status.get('severity', 0)*100:.1f}%",
                'Recovery Time': f"{status.get('recovery_time', 0)} days",
                'Capacity Impact': f"{(1 - st.session_state.supply_chain.node_capacities[node_id] / 1000)*100:.1f}%"
            })
    
    if disrupted_nodes:
        st.dataframe(pd.DataFrame(disrupted_nodes), use_container_width=True)
    else:
        st.success("✅ No current disruptions in the supply chain.")

def display_network_view():
    st.subheader("🌐 Supply Chain Network Visualization")
    
    # Create network graph visualization
    fig = create_network_visualization()
    st.plotly_chart(fig, use_container_width=True)
    
    # Node details
    st.subheader("🔍 Node Details")
    
    nodes_df, edges_df = st.session_state.supply_chain.to_dataframe()
    
    col1, col2 = st.columns(2)
    
    with col1:
        selected_node = st.selectbox(
            "Select Node",
            options=nodes_df['id'].tolist()
        )
    
    with col2:
        if selected_node:
            node_info = nodes_df[nodes_df['id'] == selected_node].iloc[0]
            st.markdown(f"""
            <div class='metric-card'>
                <h4>{selected_node}</h4>
                <p><strong>Type:</strong> {node_info['type']}</p>
                <p><strong>Capacity:</strong> {node_info['capacity']:,.0f}</p>
                <p><strong>Cost:</strong> ${node_info['cost']:,.0f}</p>
                <p><strong>Status:</strong> {'🔴 Disrupted' if node_info['disrupted'] else '🟢 Operational'}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Edge details
    st.subheader("🛣️ Connection Details")
    
    if selected_node:
        connected_edges = edges_df[
            (edges_df['source'] == selected_node) | 
            (edges_df['target'] == selected_node)
        ]
        
        if not connected_edges.empty:
            st.dataframe(connected_edges, use_container_width=True)
        else:
            st.info(f"No connections found for {selected_node}")

def create_network_visualization():
    """Create interactive network visualization"""
    nodes_df, edges_df = st.session_state.supply_chain.to_dataframe()
    
    # Create node trace
    node_trace = go.Scatter(
        x=nodes_df['x'],
        y=nodes_df['y'],
        mode='markers+text',
        text=nodes_df['id'],
        textposition="top center",
        marker=dict(
            size=nodes_df['capacity'] / 100 + 10,
            color=nodes_df['disrupted'].apply(lambda x: 'red' if x else 'green'),
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Status"),
            line=dict(width=2, color='DarkSlateGrey')
        ),
        hovertemplate='<b>%{text}</b><br>' +
                     'Type: %{customdata[0]}<br>' +
                     'Capacity: %{customdata[1]:,.0f}<br>' +
                     'Cost: $%{customdata[2]:,.0f}<br>' +
                     '<extra></extra>',
        customdata=nodes_df[['type', 'capacity', 'cost']]
    )
    
    # Create edge trace
    edge_traces = []
    for _, edge in edges_df.iterrows():
        source_node = nodes_df[nodes_df['id'] == edge['source']].iloc[0]
        target_node = nodes_df[nodes_df['id'] == edge['target']].iloc[0]
        
        edge_trace = go.Scatter(
            x=[source_node['x'], target_node['x'], None],
            y=[source_node['y'], target_node['y'], None],
            mode='lines',
            line=dict(
                width=2,
                color='red' if edge['disrupted'] else 'blue',
                dash='dash' if edge['disrupted'] else 'solid'
            ),
            hoverinfo='text',
            text=f"Capacity: {edge['capacity']}<br>Cost: ${edge['cost']}<br>Lead Time: {edge['lead_time']} days",
            opacity=0.6
        )
        edge_traces.append(edge_trace)
    
    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])
    
    fig.update_layout(
        title="Global Supply Chain Network",
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600
    )
    
    return fig

def display_analytics():
    st.subheader("📊 Supply Chain Analytics")
    
    # Robustness metrics
    robustness = st.session_state.optimizer.calculate_robustness_metrics()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Network Density", f"{robustness['network_density']:.3f}")
    
    with col2:
        st.metric("Redundancy Score", f"{robustness['redundancy_score']:.3f}")
    
    with col3:
        st.metric("Critical Nodes", len(robustness.get('critical_nodes', [])))
    
    # Critical nodes table
    if robustness.get('critical_nodes'):
        st.subheader("🎯 Critical Nodes Analysis")
        critical_df = pd.DataFrame(robustness['critical_nodes'])
        st.dataframe(critical_df, use_container_width=True)
    
    # Flow optimization
    st.subheader("📦 Flow Optimization")
    
    # Sample demand
    demand = {f"R{i}_": np.random.randint(100, 500) for i in range(1, 13)}
    
    if st.button("Optimize Material Flows"):
        with st.spinner("Optimizing flows..."):
            optimized_flows = st.session_state.optimizer.optimize_flows(demand)
            
            if optimized_flows:
                flows_df = pd.DataFrame([
                    {'From': source, 'To': target, 'Flow': flow}
                    for (source, target), flow in optimized_flows.items()
                    if flow > 0
                ])
                
                st.success(f"Optimized {len(flows_df)} flows!")
                st.dataframe(flows_df, use_container_width=True)
                
                # Visualization
                fig = px.bar(flows_df.nlargest(20, 'Flow'), 
                            x='Flow', y='From', color='To',
                            title="Top 20 Material Flows",
                            orientation='h')
                st.plotly_chart(fig, use_container_width=True)
    
    # Inventory optimization
    st.subheader("📊 Inventory Optimization")
    
    # Generate sample demand forecast
    demand_forecast = {}
    for node in st.session_state.supply_chain.graph.nodes():
        if st.session_state.supply_chain.node_types[node].value == 'retailer':
            demand_forecast[node] = np.random.normal(200, 50, 30).tolist()
    
    if st.button("Calculate Optimal Inventory"):
        with st.spinner("Calculating optimal inventory levels..."):
            optimal_inventory = st.session_state.optimizer.optimize_inventory(demand_forecast)
            
            inventory_df = pd.DataFrame([
                {'Node': node, 'Optimal Inventory': inv}
                for node, inv in optimal_inventory.items()
            ])
            
            st.dataframe(inventory_df, use_container_width=True)
            
            # Chart
            fig = px.bar(inventory_df, x='Node', y='Optimal Inventory',
                        title="Optimal Inventory Levels by Node")
            st.plotly_chart(fig, use_container_width=True)

def display_ai_insights():
    st.subheader("🤖 AI-Powered Insights")
    
    # RL Agent Status
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class='metric-card'>
            <h4>🧠 Deep RL Agent</h4>
            <p><strong>Status:</strong> Ready for Training</p>
            <p><strong>State Space:</strong> 250 dimensions</p>
            <p><strong>Action Space:</strong> 75 actions</p>
            <p><strong>Training Episodes:</strong> 0/1000</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🎯 Train RL Agent"):
            with st.spinner("Training RL agent..."):
                # Simulate training
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                st.success("RL Agent trained successfully!")
    
    with col2:
        st.markdown("""
        <div class='metric-card'>
            <h4>🔍 GNN Model</h4>
            <p><strong>Status:</strong> Active</p>
            <p><strong>Nodes Analyzed:</strong> 35</p>
            <p><strong>Edges Analyzed:</strong> 120</p>
            <p><strong>Prediction Accuracy:</strong> 92.3%</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("📈 Run GNN Analysis"):
            with st.spinner("Running graph analysis..."):
                time.sleep(2)
                st.success("GNN analysis complete!")
    
    # AI Recommendations
    st.markdown("---")
    st.subheader("💡 AI Recommendations")
    
    recommendations = [
        {
            "priority": "High",
            "recommendation": "Diversify supplier base in Southeast Asia to reduce dependency on M2_CN",
            "impact": "Resilience: +15%, Cost: +5%",
            "timeline": "3-6 months"
        },
        {
            "priority": "Medium",
            "recommendation": "Increase safety stock at DC3_CN by 30%",
            "impact": "Service Level: +8%, Inventory Cost: +12%",
            "timeline": "1 month"
        },
        {
            "priority": "Low",
            "recommendation": "Implement blockchain for shipment tracking",
            "impact": "Transparency: +25%, Cost: +3%",
            "timeline": "6-12 months"
        }
    ]
    
    for rec in recommendations:
        with st.expander(f"{rec['priority']} Priority: {rec['recommendation']}"):
            st.markdown(f"""
            **Impact:** {rec['impact']}
            
            **Implementation Timeline:** {rec['timeline']}
            """)
            
            if st.button(f"Implement", key=f"implement_{rec['priority']}"):
                st.success(f"Implementation of '{rec['recommendation']}' started!")
    
    # Simulation Control
    st.markdown("---")
    st.subheader("🎮 Simulation Control")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("▶️ Start Simulation"):
            st.info("Simulation started. Running for 30 time steps...")
    
    with col2:
        if st.button("⏸️ Pause Simulation"):
            st.info("Simulation paused.")
    
    with col3:
        if st.button("⏹️ Stop Simulation"):
            st.info("Simulation stopped.")

if __name__ == "__main__":
    main()