"""
Real-time interactive dashboard using Dash/Plotly.
Shows live navigation solution, beacon status, PDOP, etc.
"""

import dash
from dash import dcc, html, Input, Output, callback
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timezone
from typing import Dict, List, Optional
import threading
import queue

from navigation.navigator import AdvancedBeaconNavigator
from core.beacon import Beacon

class NavigationDashboard:
    """
    Dash web application for real-time navigation monitoring.
    """
    
    def __init__(self, 
                 navigator: AdvancedBeaconNavigator,
                 port: int = 8050,
                 update_interval_ms: int = 1000):
        self.navigator = navigator
        self.port = port
        self.update_interval = update_interval_ms
        self.data_queue = queue.Queue(maxsize=1000)
        self.running = False
        
        # Initialize Dash app
        self.app = dash.Dash(__name__, suppress_callback_exceptions=True)
        self._setup_layout()
        self._setup_callbacks()
        
    def _setup_layout(self):
        """Define dashboard layout"""
        self.app.layout = html.Div([
            html.H1("DSProbe Dashboard", style={'textAlign': 'center'}),
            
            # Top row: 3D visualization + metrics
            html.Div([
                dcc.Graph(id='3d-trajectory', style={'width': '60%', 'height': '600px'}),
                html.Div([
                    html.H3("Current Status"),
                    html.Table([
                        html.Tr([html.Td("Position:"), html.Td(id='pos-table')]),
                        html.Tr([html.Td("Velocity:"), html.Td(id='vel-table')]),
                        html.Tr([html.Td("PDOP:"), html.Td(id='pdop-table')]),
                        html.Tr([html.Td("Time:"), html.Td(id='time-table')]),
                        html.Tr([html.Td("Visible Beacons:"), html.Td(id='beacon-count')]),
                        html.Tr([html.Td("Measurements:"), html.Td(id='meas-count')]),
                    ], style={'width': '100%', 'fontSize': '16px'}),
                    html.Hr(),
                    html.H3("Collision Warnings"),
                    html.Div(id='collision-warnings'),
                    html.Hr(),
                    html.H3("Maneuver Suggestions"),
                    html.Div(id='maneuvers')
                ], style={'width': '35%', 'margin-left': '5%'})
            ], style={'display': 'flex'}),
            
            # Bottom row: time series
            html.Div([
                dcc.Graph(id='pdop-timeseries'),
                dcc.Graph(id='beacon-health')
            ], style={'display': 'flex'}),
            
            # Beacon status table
            html.H3("Beacon Status"),
            dash.dash_table.DataTable(
                id='beacon-table',
                columns=[
                    {'name': 'ID', 'id': 'id'},
                    {'name': 'Name', 'id': 'name'},
                    {'name': 'Type', 'id': 'type'},
                    {'name': 'Health', 'id': 'health'},
                    {'name': 'Reliability', 'id': 'reliability'},
                    {'name': 'Visible', 'id': 'visible'}
                ],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left', 'padding': '5px'}
            ),
            
            # Hidden div for storing data
            dcc.Interval(id='interval', interval=self.update_interval, n_intervals=0),
            dcc.Store(id='navigation-data')
        ])
    
    def _setup_callbacks(self):
        """Setup Dash callbacks for real-time updates"""
        
        @self.app.callback(
            Output('navigation-data', 'data'),
            Input('interval', 'n_intervals')
        )
        def update_data(n):
            """Fetch latest navigation solution"""
            if not self.navigator.current_time:
                return {}
                
            sol = self.navigator.get_solution()
            # Convert numpy to list for JSON
            sol_json = {
                'position': sol['position'].tolist(),
                'velocity': sol['velocity'].tolist(),
                'pdop': float(sol['pdop']),
                'timestamp': sol['timestamp'].isoformat(),
                'visible_beacons': sol['visible_beacons'],
                'measurements_used': sol['measurements_used'],
                'collision_warnings': sol['collision_warnings'],
                'maneuvers': [{
                    'delta_v': m.delta_v.tolist(),
                    'reason': m.reason,
                    'confidence': m.confidence
                } for m in sol['avoidance_maneuvers']]
            }
            return sol_json
        
        @self.app.callback(
            [Output('3d-trajectory', 'figure'),
             Output('pos-table', 'children'),
             Output('vel-table', 'children'),
             Output('pdop-table', 'children'),
             Output('time-table', 'children'),
             Output('beacon-count', 'children'),
             Output('meas-count', 'children')],
            Input('navigation-data', 'data')
        )
        def update_main_display(data):
            if not data:
                return {}, "", "", "", "", "", ""
                
            # 3D Figure
            fig_3d = go.Figure()
            
            # Spacecraft position
            pos = data['position']
            fig_3d.add_trace(go.Scatter3d(
                x=[pos[0]], y=[pos[1]], z=[pos[2]],
                mode='markers',
                marker=dict(size=10, color='red'),
                name='Spacecraft'
            ))
            
            # Beacons
            beacon_positions = []
            beacon_names = []
            for bid, beacon in self.navigator.beacons.items():
                bpos = beacon.get_position(0.0).tolist()
                beacon_positions.append(bpos)
                beacon_names.append(beacon.name)
                
            if beacon_positions:
                bp = np.array(beacon_positions)
                fig_3d.add_trace(go.Scatter3d(
                    x=bp[:,0], y=bp[:,1], z=bp[:,2],
                    mode='markers+text',
                    marker=dict(size=8, color='yellow', symbol='diamond'),
                    text=beacon_names,
                    textposition="top center",
                    name='Beacons'
                ))
                
            # Trajectory history
            if hasattr(self.navigator, 'state_history') and len(self.navigator.state_history) > 1:
                hist = np.array([s.position for s in self.navigator.state_history])
                fig_3d.add_trace(go.Scatter3d(
                    x=hist[:,0], y=hist[:,1], z=hist[:,2],
                    mode='lines',
                    line=dict(color='blue', width=2),
                    name='Trajectory'
                ))
                
            fig_3d.update_layout(
                scene=dict(
                    xaxis_title='X (km)',
                    yaxis_title='Y (km)',
                    zaxis_title='Z (km)',
                    aspectmode='cube'
                ),
                margin=dict(l=0, r=0, b=0, t=30)
            )
            
            # Tables
            pos_str = f"X: {pos[0]/1e3:.1f}, Y: {pos[1]/1e3:.1f}, Z: {pos[2]/1e3:.1f} km"
            vel = data['velocity']
            vel_str = f"Vx: {vel[0]:.3f}, Vy: {vel[1]:.3f}, Vz: {vel[2]:.3f} km/s"
            pdop_str = f"{data['pdop']:.2f} km"
            time_str = data['timestamp'][:19] if 'timestamp' in data else ""
            beacon_cnt = str(data['visible_beacons'])
            meas_cnt = str(data['measurements_used'])
            
            return fig_3d, pos_str, vel_str, pdop_str, time_str, beacon_cnt, meas_cnt
        
        @self.app.callback(
            [Output('collision-warnings', 'children'),
             Output('maneuvers', 'children')],
            Input('navigation-data', 'data')
        )
        def update_warnings(data):
            if not data:
                return "", ""
                
            # Collision warnings
            warnings_list = []
            for warning in data.get('collision_warnings', []):
                warnings_list.append(html.P(f"⚠️ {warning}", style={'color': 'orange'}))
                
            # Maneuvers
            maneuvers_list = []
            for m in data.get('maneuvers', []):
                dv = np.array(m['delta_v'])
                maneuvers_list.append(html.P(
                    f"🚀 ΔV: {np.linalg.norm(dv)*1000:.1f} m/s - {m['reason']}",
                    style={'color': 'green'}
                ))
                
            return warnings_list, maneuvers_list
        
        @self.app.callback(
            [Output('pdop-timeseries', 'figure'),
             Output('beacon-health', 'figure')],
            Input('navigation-data', 'data')
        )
        def update_timeseries(data):
            # PDOP time series (use stored history)
            if hasattr(self.navigator, 'pdop_history'):
                fig_pdop = go.Figure()
                fig_pdop.add_trace(go.Scatter(
                    y=self.navigator.pdop_history,
                    mode='lines',
                    name='PDOP'
                ))
                fig_pdop.update_layout(
                    title='PDOP Over Time',
                    yaxis_title='PDOP (km)',
                    height=300
                )
            else:
                fig_pdop = go.Figure()
                
            # Beacon health bar chart
            beacon_ids = []
            healths = []
            for bid, beacon in self.navigator.beacons.items():
                beacon_ids.append(beacon.name)
                healths.append(beacon.health)
                
            if beacon_ids:
                fig_health = go.Figure()
                fig_health.add_trace(go.Bar(
                    x=beacon_ids,
                    y=healths,
                    marker_color=['green' if h>0.7 else 'orange' if h>0.3 else 'red' for h in healths]
                ))
                fig_health.update_layout(
                    title='Beacon Health',
                    yaxis_title='Health (0-1)',
                    yaxis_range=[0,1],
                    height=300
                )
            else:
                fig_health = go.Figure()
                
            return fig_pdop, fig_health
        
        @self.app.callback(
            Output('beacon-table', 'data'),
            Input('navigation-data', 'data')
        )
        def update_beacon_table(data):
            rows = []
            for bid, beacon in self.navigator.beacons.items():
                rows.append({
                    'id': bid,
                    'name': beacon.name,
                    'type': beacon.beacon_type.value,
                    'health': f"{beacon.health:.2f}",
                    'reliability': f"{beacon.reliability:.2f}",
                    'visible': '✓' if bid in data.get('visible_beacons', []) else '✗'
                })
            return rows
    
    def start(self, debug: bool = False, use_reloader: bool = False):
        """Start dashboard server"""
        print(f"Starting dashboard on http://localhost:{self.port}")
        self.running = True
        self.app.run_server(debug=debug, use_reloader=use_reloader, port=self.port)
        
    def start_background(self):
        """Start dashboard in background thread"""
        thread = threading.Thread(target=self.start, args=(False, False))
        thread.daemon = True
        thread.start()
        return thread
        
    def add_data_point(self, 
                      position: np.ndarray,
                      pdop: float,
                      timestamp: datetime,
                      measurements_used: int):
        """Add data point for plotting (if using queue-based)"""
        try:
            self.data_queue.put_nowait({
                'position': position,
                'pdop': pdop,
                'timestamp': timestamp,
                'measurements': measurements_used
            })
        except queue.Full:
            pass  # drop oldest?
            
    def stop(self):
        """Stop dashboard"""
        self.running = False

def launch_dashboard(navigator: AdvancedBeaconNavigator, 
                    port: int = 8050,
                    blocking: bool = True) -> NavigationDashboard:
    """
    Convenience function to launch dashboard.
    
    Args:
        navigator: The navigation system to monitor
        port: Web port
        blocking: If True, blocks; if False, runs in background thread
        
    Returns:
        NavigationDashboard instance
    """
    dashboard = NavigationDashboard(navigator, port=port)
    if blocking:
        dashboard.start()
    else:
        dashboard.start_background()
    return dashboard

# Standalone test
if __name__ == "__main__":
    # Quick demo with dummy navigator
    from config.settings import NavConfig
    from core.beacon import Beacon, create_planet_ephemeris
    
    # Create dummy system (would need full navigator setup)
    # dashboard = launch_dashboard(navigator)
    print("Dashboard module loaded. Integrate with AdvancedBeaconNavigator and call launch_dashboard().")