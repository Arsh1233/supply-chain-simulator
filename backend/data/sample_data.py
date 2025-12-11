import numpy as np
import pandas as pd
from backend.models.supply_chain_graph import SupplyChainGraph, NodeType
import random

def create_sample_supply_chain() -> SupplyChainGraph:
    """Create a sample global supply chain"""
    sc = SupplyChainGraph()
    
    # Create suppliers (10 suppliers worldwide)
    suppliers = [
        ("S1_CN", (121.47, 31.23)),  # Shanghai
        ("S2_US", (-74.01, 40.71)),  # New York
        ("S3_DE", (13.40, 52.52)),   # Berlin
        ("S4_JP", (139.69, 35.69)),  # Tokyo
        ("S5_IN", (77.21, 28.61)),   # Delhi
        ("S6_BR", (-46.63, -23.55)), # Sao Paulo
        ("S7_RU", (37.62, 55.75)),   # Moscow
        ("S8_AU", (151.21, -33.87)), # Sydney
        ("S9_KR", (126.98, 37.57)),  # Seoul
        ("S10_SG", (103.85, 1.28))   # Singapore
    ]
    
    for sid, loc in suppliers:
        sc.add_node(
            node_id=sid,
            node_type=NodeType.SUPPLIER,
            capacity=random.uniform(1000, 5000),
            cost=random.uniform(10, 50),
            location=loc,
            reliability=random.uniform(0.85, 0.98)
        )
    
    # Create manufacturers (5 manufacturers)
    manufacturers = [
        ("M1_US", (-118.24, 34.05)),  # Los Angeles
        ("M2_CN", (113.27, 23.13)),   # Guangzhou
        ("M3_DE", (6.96, 50.94)),     # Cologne
        ("M4_MX", (-99.13, 19.43)),   # Mexico City
        ("M5_IN", (72.88, 19.08))     # Mumbai
    ]
    
    for mid, loc in manufacturers:
        sc.add_node(
            node_id=mid,
            node_type=NodeType.MANUFACTURER,
            capacity=random.uniform(2000, 8000),
            cost=random.uniform(20, 80),
            location=loc,
            reliability=random.uniform(0.9, 0.99)
        )
    
    # Create distribution centers (8 DCs)
    dcs = [
        ("DC1_US", (-87.62, 41.88)),  # Chicago
        ("DC2_EU", (2.35, 48.86)),    # Paris
        ("DC3_CN", (116.41, 39.90)),  # Beijing
        ("DC4_AE", (55.27, 25.20)),   # Dubai
        ("DC5_UK", (-0.13, 51.51)),   # London
        ("DC6_SG", (103.85, 1.28)),   # Singapore
        ("DC7_JP", (135.50, 34.69)),  # Osaka
        ("DC8_AU", (144.96, -37.81))  # Melbourne
    ]
    
    for dcid, loc in dcs:
        sc.add_node(
            node_id=dcid,
            node_type=NodeType.DISTRIBUTION_CENTER,
            capacity=random.uniform(1500, 6000),
            cost=random.uniform(15, 60),
            location=loc,
            reliability=random.uniform(0.92, 0.97)
        )
    
    # Create retailers (12 retailers)
    retailers = [
        ("R1_NYC", (-74.01, 40.71)),
        ("R2_LA", (-118.24, 34.05)),
        ("R3_LON", (-0.13, 51.51)),
        ("R4_PAR", (2.35, 48.86)),
        ("R5_TOK", (139.69, 35.69)),
        ("R6_SHA", (121.47, 31.23)),
        ("R7_SYD", (151.21, -33.87)),
        ("R8_DXB", (55.27, 25.20)),
        ("R9_SIN", (103.85, 1.28)),
        ("R10_MUM", (72.88, 19.08)),
        ("R11_SAO", (-46.63, -23.55)),
        ("R12_JKT", (106.85, -6.21))
    ]
    
    for rid, loc in retailers:
        sc.add_node(
            node_id=rid,
            node_type=NodeType.RETAILER,
            capacity=random.uniform(500, 2000),
            cost=random.uniform(5, 30),
            location=loc,
            reliability=random.uniform(0.88, 0.95)
        )
    
    # Create connections (edges)
    # Suppliers to Manufacturers
    for sid, _ in suppliers:
        for mid, _ in manufacturers:
            if random.random() < 0.4:  # 40% connection probability
                sc.add_edge(
                    source=sid,
                    target=mid,
                    capacity=random.uniform(500, 2000),
                    lead_time=random.randint(7, 30),
                    cost_per_unit=random.uniform(2, 8),
                    reliability=random.uniform(0.85, 0.96)
                )
    
    # Manufacturers to DCs
    for mid, _ in manufacturers:
        for dcid, _ in dcs:
            if random.random() < 0.5:
                sc.add_edge(
                    source=mid,
                    target=dcid,
                    capacity=random.uniform(1000, 4000),
                    lead_time=random.randint(3, 14),
                    cost_per_unit=random.uniform(1, 5),
                    reliability=random.uniform(0.88, 0.98)
                )
    
    # DCs to Retailers
    for dcid, _ in dcs:
        for rid, _ in retailers:
            if random.random() < 0.6:
                sc.add_edge(
                    source=dcid,
                    target=rid,
                    capacity=random.uniform(300, 1500),
                    lead_time=random.randint(1, 7),
                    cost_per_unit=random.uniform(0.5, 3),
                    reliability=random.uniform(0.9, 0.99)
                )
    
    # Add some cross-DC transfers for redundancy
    for i in range(len(dcs)):
        for j in range(i+1, len(dcs)):
            if random.random() < 0.3:
                sc.add_edge(
                    source=dcs[i][0],
                    target=dcs[j][0],
                    capacity=random.uniform(200, 1000),
                    lead_time=random.randint(2, 10),
                    cost_per_unit=random.uniform(1.5, 6),
                    reliability=random.uniform(0.85, 0.95)
                )
    
    return sc

def generate_demand_data(num_periods: int = 30) -> pd.DataFrame:
    """Generate sample demand data"""
    periods = pd.date_range(start='2024-01-01', periods=num_periods, freq='D')
    
    demand_data = {'period': periods}
    
    # Generate demand for each retailer
    retailers = [f"R{i}_" for i in range(1, 13)]
    for retailer in retailers:
        # Base demand with trend and seasonality
        base = np.random.normal(100, 20, num_periods)
        trend = np.linspace(0, 0.5, num_periods)
        seasonality = 20 * np.sin(2 * np.pi * np.arange(num_periods) / 7)
        
        demand = base * (1 + trend) + seasonality
        # Add some random spikes
        for i in range(num_periods):
            if np.random.random() < 0.05:  # 5% chance of spike
                demand[i] *= np.random.uniform(1.5, 3.0)
        
        demand_data[f"Demand_{retailer}"] = np.maximum(demand, 10)
    
    return pd.DataFrame(demand_data)

def create_disruption_scenario(sc: SupplyChainGraph):
    """Apply sample disruptions to supply chain"""
    # Disrupt a major manufacturer
    sc.apply_disruption("M2_CN", severity=0.7, duration=30)
    
    # Disrupt key transportation route
    sc.apply_edge_disruption("S1_CN", "M2_CN", severity=0.8, duration=14)
    
    # Disrupt a distribution center
    sc.apply_disruption("DC3_CN", severity=0.5, duration=21)
    
    # Disrupt trans-pacific shipping
    sc.apply_edge_disruption("DC6_SG", "DC1_US", severity=0.6, duration=28)