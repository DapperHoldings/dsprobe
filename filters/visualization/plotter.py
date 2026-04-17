"""
Matplotlib-based plotting utilities for navigation.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple, Optional, Dict
import matplotlib.animation as animation

from core.beacon import Beacon
from core.state import State
from utils.geometry import compute_pdop

class NavigationPlotter:
    """
    3D visualization of spacecraft trajectory, beacons, and navigation metrics.
    """
    
    def __init__(self, 
                 figsize: Tuple[int, int] = (12, 10),
                 dark_mode: bool = False):
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        if dark_mode:
            plt.style.use('dark_background')
            self.fig.patch.set_facecolor('black')
            self.ax.set_facecolor('black')
            
        # Trajectory
        self.trajectory_history = []
        self.uncertainty_ellipses = []
        
        # Beacon plots
        self.beacon_scatter = None
        self.beacon_labels = {}
        
        # Metrics
        self.pdop_history = []
        self.time_history = []
        
    def set_beacons(self, beacons: List[Beacon]):
        """Plot beacon positions"""
        positions = np.array([b.get_position(0.0) for b in beacons])
        self.ax.scatter(positions[:,0], positions[:,1], positions[:,2], 
                       c='yellow', s=100, marker='*', label='Beacons', alpha=0.8)
        
        for i, b in enumerate(beacons):
            self.ax.text(positions[i,0], positions[i,1], positions[i,2], 
                        b.name, fontsize=8, color='yellow')
            
    def update_trajectory(self, position: np.ndarray, 
                         covariance: Optional[np.ndarray] = None):
        """Add new position to trajectory"""
        self.trajectory_history.append(position.copy())
        
        # Plot trajectory line
        traj = np.array(self.trajectory_history)
        if len(traj) > 1:
            self.ax.plot(traj[:,0], traj[:,1], traj[:,2], 
                        'b-', linewidth=1, alpha=0.7, label='Trajectory')
            
        # Plot current position
        self.ax.scatter([position[0]], [position[1]], [position[2]], 
                       c='red', s=50, marker='o', label='Spacecraft')
        
        # Plot uncertainty ellipsoid (3-sigma)
        if covariance is not None:
            self._plot_ellipsoid(position, covariance[0:3,0:3])
            
    def _plot_ellipsoid(self, center: np.ndarray, cov: np.ndarray, 
                       n_std: float = 3.0):
        """Plot 3D ellipsoid representing covariance"""
        # Eigen decomposition
        eigenvals, eigenvecs = np.linalg.eigh(cov)
        # Sort
        idx = eigenvals.argsort()[::-1]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Generate points on sphere
        u = np.linspace(0, 2*np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = np.outer(np.cos(u), np.sin(v))
        y = np.outer(np.sin(u), np.sin(v))
        z = np.outer(np.ones_like(u), np.cos(v))
        
        # Scale by eigenvalues and rotate
        points = np.stack([x, y, z], axis=-1)  # (20,20,3)
        scales = np.sqrt(eigenvals) * n_std
        points = points * scales
        points = points @ eigenvecs.T + center
        
        # Plot surface
        self.ax.plot_surface(points[:,:,0], points[:,:,1], points[:,:,2],
                            alpha=0.1, color='red')
        
    def update_pdop(self, pdop: float, timestamp: float):
        """Update PDOP history"""
        self.pdop_history.append(pdop)
        self.time_history.append(timestamp)
        
    def plot_pdop_timeseries(self, ax=None):
        """Plot PDOP over time"""
        if ax is None:
            fig, ax = plt.subplots()
        ax.plot(self.time_history, self.pdop_history, 'g-')
        ax.set_xlabel('Time')
        ax.set_ylabel('PDOP (km)')
        ax.set_title('Navigation Accuracy Over Time')
        ax.grid(True, alpha=0.3)
        return ax
    
    def plot_beacon_visibility(self, 
                              beacon: Beacon,
                              observer_positions: np.ndarray,
                              times: np.ndarray):
        """Plot beacon visibility over time"""
        elevations = []
        for pos in observer_positions:
            b_pos = beacon.get_position(0.0)
            vec = b_pos - pos
            r = np.linalg.norm(vec)
            # Elevation relative to local horizontal? Simplified: just range
            elevations.append(r)
            
        fig, ax = plt.subplots()
        ax.plot(times, elevations)
        ax.set_xlabel('Time')
        ax.set_ylabel(f'Range to {beacon.name} (km)')
        ax.set_title(f'Visibility of {beacon.name}')
        ax.grid(True, alpha=0.3)
        return fig, ax
    
    def animate_trajectory(self, 
                          trajectory: List[np.ndarray],
                          beacons: List[Beacon],
                          interval: int = 100):
        """Create animation of trajectory"""
        fig = plt.figure(figsize=(10,8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot static beacons
        b_pos = np.array([b.get_position(0.0) for b in beacons])
        ax.scatter(b_pos[:,0], b_pos[:,1], b_pos[:,2], c='yellow', s=100, marker='*')
        
        # Trajectory line
        line, = ax.plot([], [], [], 'b-')
        point, = ax.plot([], [], [], 'ro')
        
        def init():
            line.set_data([], [])
            line.set_3d_properties([])
            point.set_data([], [])
            point.set_3d_properties([])
            return line, point
        
        def update(frame):
            traj = np.array(trajectory[:frame+1])
            line.set_data(traj[:,0], traj[:,1])
            line.set_3d_properties(traj[:,2])
            point.set_data([traj[-1,0]], [traj[-1,1]])
            point.set_3d_properties([traj[-1,2]])
            return line, point
        
        ani = animation.FuncAnimation(fig, update, frames=len(trajectory),
                                     init_func=init, blit=True, interval=interval)
        return ani
    
    def show(self):
        """Display plot"""
        self.ax.set_xlabel('X (km)')
        self.ax.set_ylabel('Y (km)')
        self.ax.set_zlabel('Z (km)')
        self.ax.set_title('Spacecraft Navigation')
        self.ax.legend()
        plt.tight_layout()
        plt.show()

def plot_beacon_coverage(beacons: List[Beacon], 
                        grid_resolution: int = 50,
                        region: Tuple[float, float, float, float, float, float] = (-1e5, 1e5, -1e5, 1e5, -1e5, 1e5)):
    """
    Plot heatmap of PDOP across a 3D region (projected to 2D slices).
    """
    x_min, x_max, y_min, y_max, z_min, z_max = region
    x = np.linspace(x_min, x_max, grid_resolution)
    y = np.linspace(y_min, y_max, grid_resolution)
    X, Y = np.meshgrid(x, y)
    
    # Compute PDOP at each (x,y) at fixed z=0
    Z = np.zeros_like(X)
    for i in range(grid_resolution):
        for j in range(grid_resolution):
            pos = np.array([X[i,j], Y[i,j], 0])
            # Compute Fisher information from all beacons
            H = []
            for beacon in beacons:
                b_pos = beacon.get_position(0.0)
                r_vec = pos - b_pos
                r = np.linalg.norm(r_vec)
                if r < 1e-6:
                    continue
                r_hat = r_vec / r
                # Range measurement contribution
                H.append(r_hat)
                # Direction contribution (2 independent from 3-vector minus 1 constraint)
                # Actually direction gives 2 DOF; we'll approximate with full 3 then handle later
                # Simpler: just use range info for PDOP
            if len(H) < 3:
                Z[i,j] = np.inf
            else:
                H = np.array(H)
                I = H.T @ H
                try:
                    cov = np.linalg.inv(I)
                    Z[i,j] = np.sqrt(np.trace(cov[0:3,0:3]))
                except:
                    Z[i,j] = np.inf
                    
    fig, ax = plt.subplots(figsize=(10,8))
    contour = ax.contourf(X, Y, Z, levels=50, cmap='viridis', 
                         norm=plt.Normalize(vmin=0, vmax=1000))
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_title('PDOP Heatmap (Z=0 plane)')
    fig.colorbar(contour, label='PDOP (km)')
    return fig, ax

def plot_pdop_map(beacons: List[Beacon], 
                  altitude: float = 0.0,
                  resolution: int = 100,
                  radius: float = 1e5):
    """
    2D polar plot of PDOP around a planet.
    """
    theta = np.linspace(0, 2*np.pi, resolution)
    r = np.linspace(0, radius, resolution)
    Theta, R = np.meshgrid(theta, r)
    PDOP = np.zeros_like(Theta)
    
    planet_center = beacons[0].get_position(0.0) if beacons else np.zeros(3)
    
    for i in range(resolution):
        for j in range(resolution):
            # Convert polar to ECEF-like
            x = planet_center[0] + R[i,j] * np.cos(Theta[i,j])
            y = planet_center[1] + R[i,j] * np.sin(Theta[i,j])
            z = planet_center[2] + altitude
            pos = np.array([x,y,z])
            
            # Compute PDOP from beacons
            H = []
            for beacon in beacons[1:]:  # skip central body
                bpos = beacon.get_position(0.0)
                vec = pos - bpos
                dist = np.linalg.norm(vec)
                if dist > 1e-6:
                    H.append(vec/dist)
                    
            if len(H) >= 3:
                H = np.array(H)
                I = H.T @ H
                try:
                    cov = np.linalg.inv(I)
                    PDOP[i,j] = np.sqrt(np.trace(cov))
                except:
                    PDOP[i,j] = np.inf
            else:
                PDOP[i,j] = np.inf
                
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(8,8))
    c = ax.pcolormesh(Theta, R/1000, PDOP, shading='auto', cmap='RdYlBu_r', 
                     norm=plt.Normalize(vmin=0, vmax=500))
    ax.set_title('PDOP around planet (km)')
    fig.colorbar(c, label='PDOP (km)')
    return fig, ax