"""
Dynamic Rainfall Visualization for Hong Kong Observatory Data
============================================================

This module creates real-time, artistic visualizations of Hong Kong rainfall data
using matplotlib with flowing visual patterns and dynamic color landscapes.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__)))

try:
    from data_fetcher import HKORainfallDataFetcher, RealTimeDataStream
    from data_processor import RainfallDataProcessor
except ImportError:
    print("Note: Data modules not available in current environment. Using fallback methods.")


class ArtisticRainfallVisualizer:
    """
    Creates artistic, dynamic visualizations of Hong Kong rainfall data.
    """
    
    def __init__(self, style='artistic', figsize=(16, 12)):
        """
        Initialize the visualizer.
        
        Args:
            style: Visualization style ('artistic', 'scientific', 'abstract')
            figsize: Figure size tuple
        """
        self.style = style
        self.figsize = figsize
        
        # Set up matplotlib for high-quality output
        plt.style.use('dark_background')
        self.fig = None
        self.axes = None
        
        # Initialize data components
        try:
            self.data_fetcher = HKORainfallDataFetcher()
            self.data_processor = RainfallDataProcessor()
        except:
            print("Warning: Data fetcher/processor not available. Using simulation mode.")
            self.data_fetcher = None
            self.data_processor = None
        
        # Color schemes for different rainfall intensities
        self.color_schemes = {
            'artistic': {
                'background': '#0a0a0a',
                'colors': ['#000033', '#000066', '#003366', '#0066cc', '#00ccff', 
                          '#66ffcc', '#ccff66', '#ffcc00', '#ff6600', '#ff0000'],
                'accent': '#ff6b9d'
            },
            'natural': {
                'background': '#001122',
                'colors': ['#001122', '#002244', '#004466', '#0088aa', '#00aacc',
                          '#44ccee', '#88eeaa', '#ccee66', '#eeaa44', '#ee6644'],
                'accent': '#ffffff'
            },
            'neon': {
                'background': '#000000',
                'colors': ['#000000', '#330066', '#660066', '#990099', '#cc00cc',
                          '#ff00ff', '#ff33ff', '#ff66ff', '#ff99ff', '#ffffff'],
                'accent': '#00ffff'
            }
        }
        
        # Animation parameters
        self.animation_speed = 50  # milliseconds
        self.particle_count = 500
        self.flow_trails = []
        self.time_step = 0
        
        # Hong Kong geographical bounds
        self.hk_bounds = {
            'lat_min': 22.15, 'lat_max': 22.58,
            'lon_min': 113.83, 'lon_max': 114.41
        }
        
    def create_custom_colormap(self, scheme_name: str = 'artistic') -> LinearSegmentedColormap:
        """
        Create custom colormap for rainfall visualization.
        
        Args:
            scheme_name: Name of color scheme to use
            
        Returns:
            Custom colormap
        """
        scheme = self.color_schemes.get(scheme_name, self.color_schemes['artistic'])
        colors = scheme['colors']
        
        # Create colormap
        n_bins = len(colors)
        cmap = LinearSegmentedColormap.from_list(
            f'rainfall_{scheme_name}', colors, N=n_bins
        )
        
        return cmap
    
    def generate_sample_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate sample rainfall data for demonstration.
        
        Returns:
            Tuple of (lon_mesh, lat_mesh, rainfall_grid)
        """
        # Create coordinate meshes
        lon = np.linspace(self.hk_bounds['lon_min'], self.hk_bounds['lon_max'], 50)
        lat = np.linspace(self.hk_bounds['lat_min'], self.hk_bounds['lat_max'], 50)
        lon_mesh, lat_mesh = np.meshgrid(lon, lat)
        
        # Generate dynamic rainfall pattern
        time_factor = time.time() / 10
        
        # Create multiple storm centers
        storm1_x = 114.1 + 0.1 * np.sin(time_factor)
        storm1_y = 22.3 + 0.05 * np.cos(time_factor)
        storm2_x = 114.25 + 0.08 * np.cos(time_factor * 1.5)
        storm2_y = 22.4 + 0.06 * np.sin(time_factor * 1.2)
        
        # Calculate distances from storm centers
        dist1 = np.sqrt((lon_mesh - storm1_x)**2 + (lat_mesh - storm1_y)**2)
        dist2 = np.sqrt((lon_mesh - storm2_x)**2 + (lat_mesh - storm2_y)**2)
        
        # Create rainfall pattern with multiple storms
        rainfall1 = 25 * np.exp(-dist1 * 80) * (1 + 0.3 * np.sin(time_factor * 3))
        rainfall2 = 15 * np.exp(-dist2 * 100) * (1 + 0.2 * np.cos(time_factor * 2))
        
        # Add background precipitation
        background = 2 * np.sin(lon_mesh * 20) * np.cos(lat_mesh * 25) * np.sin(time_factor)
        background = np.maximum(background, 0)
        
        # Combine all rainfall sources
        rainfall_grid = rainfall1 + rainfall2 + background
        rainfall_grid = np.maximum(rainfall_grid, 0)  # Ensure non-negative
        
        return lon_mesh, lat_mesh, rainfall_grid
    
    def generate_flow_field(self, lon_mesh: np.ndarray, lat_mesh: np.ndarray, 
                           rainfall_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate flow field for particle animation.
        
        Args:
            lon_mesh: Longitude coordinate mesh
            lat_mesh: Latitude coordinate mesh
            rainfall_grid: Rainfall intensity grid
            
        Returns:
            Tuple of (flow_u, flow_v) velocity components
        """
        # Calculate gradients for flow direction
        grad_y, grad_x = np.gradient(rainfall_grid)
        
        # Create flow field
        time_factor = time.time() / 5
        
        # Primary flow based on rainfall gradients
        flow_u = -grad_x * 0.1 + 0.02 * np.sin(time_factor + lat_mesh * 10)
        flow_v = -grad_y * 0.1 + 0.02 * np.cos(time_factor + lon_mesh * 10)
        
        # Add swirling patterns
        swirl_strength = rainfall_grid / 20
        swirl_u = swirl_strength * np.sin(time_factor * 2 + lon_mesh * 50)
        swirl_v = swirl_strength * np.cos(time_factor * 2 + lat_mesh * 50)
        
        flow_u += swirl_u
        flow_v += swirl_v
        
        return flow_u, flow_v
    
    def create_particle_system(self, n_particles: int = 500) -> Dict:
        """
        Create particle system for flow visualization.
        
        Args:
            n_particles: Number of particles
            
        Returns:
            Dictionary with particle properties
        """
        particles = {
            'x': np.random.uniform(self.hk_bounds['lon_min'], self.hk_bounds['lon_max'], n_particles),
            'y': np.random.uniform(self.hk_bounds['lat_min'], self.hk_bounds['lat_max'], n_particles),
            'age': np.random.uniform(0, 100, n_particles),
            'intensity': np.random.uniform(0.1, 1.0, n_particles),
            'trails_x': [[] for _ in range(n_particles)],
            'trails_y': [[] for _ in range(n_particles)]
        }
        return particles
    
    def update_particles(self, particles: Dict, flow_u: np.ndarray, flow_v: np.ndarray,
                        lon_mesh: np.ndarray, lat_mesh: np.ndarray, rainfall_grid: np.ndarray):
        """
        Update particle positions based on flow field.
        
        Args:
            particles: Particle system dictionary
            flow_u, flow_v: Flow velocity components
            lon_mesh, lat_mesh: Coordinate meshes
            rainfall_grid: Rainfall intensity grid
        """
        # Interpolate flow field at particle positions
        from scipy.interpolate import RegularGridInterpolator
        
        try:
            # Create interpolators
            lon_coords = lon_mesh[0, :]
            lat_coords = lat_mesh[:, 0]
            
            interp_u = RegularGridInterpolator((lat_coords, lon_coords), flow_u, 
                                             bounds_error=False, fill_value=0)
            interp_v = RegularGridInterpolator((lat_coords, lon_coords), flow_v, 
                                             bounds_error=False, fill_value=0)
            interp_rain = RegularGridInterpolator((lat_coords, lon_coords), rainfall_grid,
                                                bounds_error=False, fill_value=0)
            
            # Update particle positions
            points = np.column_stack([particles['y'], particles['x']])
            
            u_vals = interp_u(points)
            v_vals = interp_v(points)
            rain_vals = interp_rain(points)
            
            # Move particles
            dt = 0.01
            particles['x'] += u_vals * dt
            particles['y'] += v_vals * dt
            
            # Update particle properties
            particles['age'] += 1
            particles['intensity'] = rain_vals / np.max(rainfall_grid) if np.max(rainfall_grid) > 0 else 0.1
            
        except ImportError:
            # Fallback: simple random walk with rainfall influence
            dt = 0.001
            for i in range(len(particles['x'])):
                # Add some randomness
                particles['x'][i] += np.random.normal(0, dt)
                particles['y'][i] += np.random.normal(0, dt)
        
        # Handle boundary conditions - wrap around or reset
        for i in range(len(particles['x'])):
            if (particles['x'][i] < self.hk_bounds['lon_min'] or 
                particles['x'][i] > self.hk_bounds['lon_max'] or
                particles['y'][i] < self.hk_bounds['lat_min'] or 
                particles['y'][i] > self.hk_bounds['lat_max'] or
                particles['age'][i] > 200):
                
                # Reset particle
                particles['x'][i] = np.random.uniform(self.hk_bounds['lon_min'], 
                                                    self.hk_bounds['lon_max'])
                particles['y'][i] = np.random.uniform(self.hk_bounds['lat_min'], 
                                                    self.hk_bounds['lat_max'])
                particles['age'][i] = 0
                particles['trails_x'][i] = []
                particles['trails_y'][i] = []
            
            # Update trails
            particles['trails_x'][i].append(particles['x'][i])
            particles['trails_y'][i].append(particles['y'][i])
            
            # Limit trail length
            if len(particles['trails_x'][i]) > 50:
                particles['trails_x'][i].pop(0)
                particles['trails_y'][i].pop(0)
    
    def create_static_visualization(self, save_path: str = None) -> plt.Figure:
        """
        Create a static artistic rainfall visualization.
        
        Args:
            save_path: Optional path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        # Generate data
        lon_mesh, lat_mesh, rainfall_grid = self.generate_sample_data()
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, facecolor='black')
        ax.set_facecolor('black')
        
        # Create custom colormap
        cmap = self.create_custom_colormap(self.style)
        
        # Plot rainfall as filled contours
        levels = np.linspace(0, np.max(rainfall_grid), 20)
        rainfall_plot = ax.contourf(lon_mesh, lat_mesh, rainfall_grid, 
                                  levels=levels, cmap=cmap, alpha=0.8)
        
        # Add contour lines
        contours = ax.contour(lon_mesh, lat_mesh, rainfall_grid, 
                            levels=levels[::2], colors='white', alpha=0.3, linewidths=0.5)
        
        # Generate and plot flow field
        flow_u, flow_v = self.generate_flow_field(lon_mesh, lat_mesh, rainfall_grid)
        
        # Subsample for quiver plot
        skip = 3
        ax.quiver(lon_mesh[::skip, ::skip], lat_mesh[::skip, ::skip],
                 flow_u[::skip, ::skip], flow_v[::skip, ::skip],
                 scale=2, color='cyan', alpha=0.4, width=0.002)
        
        # Styling
        ax.set_xlim(self.hk_bounds['lon_min'], self.hk_bounds['lon_max'])
        ax.set_ylim(self.hk_bounds['lat_min'], self.hk_bounds['lat_max'])
        ax.set_aspect('equal')
        
        # Remove axes for artistic effect
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add title and timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ax.set_title(f'Hong Kong Rainfall Patterns - {timestamp}', 
                    color='white', fontsize=16, pad=20)
        
        # Add colorbar
        cbar = plt.colorbar(rainfall_plot, ax=ax, shrink=0.8, aspect=30)
        cbar.set_label('Rainfall Intensity (mm/hr)', color='white', fontsize=12)
        cbar.ax.tick_params(colors='white')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor='black', edgecolor='none')
            print(f"Visualization saved to {save_path}")
        
        return fig
    
    def create_animated_visualization(self, duration: int = 60, save_path: str = None):
        """
        Create animated rainfall visualization with flowing particles.
        
        Args:
            duration: Animation duration in seconds
            save_path: Optional path to save animation as GIF
        """
        # Create figure and axis
        fig, ax = plt.subplots(figsize=self.figsize, facecolor='black')
        ax.set_facecolor('black')
        
        # Initialize particle system
        particles = self.create_particle_system(self.particle_count)
        
        # Create custom colormap
        cmap = self.create_custom_colormap(self.style)
        
        # Animation function
        def animate(frame):
            ax.clear()
            ax.set_facecolor('black')
            
            # Generate current rainfall data
            lon_mesh, lat_mesh, rainfall_grid = self.generate_sample_data()
            flow_u, flow_v = self.generate_flow_field(lon_mesh, lat_mesh, rainfall_grid)
            
            # Update particles
            self.update_particles(particles, flow_u, flow_v, lon_mesh, lat_mesh, rainfall_grid)
            
            # Plot rainfall background
            levels = np.linspace(0, np.max(rainfall_grid) if np.max(rainfall_grid) > 0 else 1, 15)
            rainfall_plot = ax.contourf(lon_mesh, lat_mesh, rainfall_grid, 
                                      levels=levels, cmap=cmap, alpha=0.6)
            
            # Plot particle trails
            for i in range(len(particles['x'])):
                if len(particles['trails_x'][i]) > 1:
                    trail_alpha = particles['intensity'][i] * 0.7
                    ax.plot(particles['trails_x'][i], particles['trails_y'][i], 
                           color='cyan', alpha=trail_alpha, linewidth=0.5)
            
            # Plot current particle positions
            sizes = particles['intensity'] * 20
            colors = particles['intensity']
            scatter = ax.scatter(particles['x'], particles['y'], s=sizes, c=colors,
                               cmap='plasma', alpha=0.8, edgecolors='white', linewidths=0.5)
            
            # Styling
            ax.set_xlim(self.hk_bounds['lon_min'], self.hk_bounds['lon_max'])
            ax.set_ylim(self.hk_bounds['lat_min'], self.hk_bounds['lat_max'])
            ax.set_aspect('equal')
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add dynamic title
            timestamp = datetime.now().strftime("%H:%M:%S")
            ax.set_title(f'Hong Kong Rainfall Flow - Live {timestamp}', 
                        color='white', fontsize=14, pad=20)
            
            return rainfall_plot, scatter
        
        # Create animation
        frames = duration * 1000 // self.animation_speed
        anim = animation.FuncAnimation(fig, animate, frames=frames, 
                                     interval=self.animation_speed, blit=False)
        
        if save_path:
            print(f"Saving animation to {save_path}...")
            anim.save(save_path, writer='pillow', fps=20, dpi=150)
            print("Animation saved!")
        
        plt.show()
        return anim


class RealTimeVisualizer:
    """
    Real-time visualization that updates with live data.
    """
    
    def __init__(self):
        self.visualizer = ArtisticRainfallVisualizer()
        self.update_interval = 5000  # 5 seconds
        
    def start_realtime_display(self):
        """
        Start real-time visualization display.
        """
        print("Starting real-time Hong Kong rainfall visualization...")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(16, 10), facecolor='black')
        ax.set_facecolor('black')
        
        def update_display():
            """Update the display with new data."""
            try:
                # Generate new data (replace with real data fetching)
                lon_mesh, lat_mesh, rainfall_grid = self.visualizer.generate_sample_data()
                
                ax.clear()
                ax.set_facecolor('black')
                
                # Create visualization
                cmap = self.visualizer.create_custom_colormap()
                levels = np.linspace(0, np.max(rainfall_grid), 20)
                
                rainfall_plot = ax.contourf(lon_mesh, lat_mesh, rainfall_grid,
                                          levels=levels, cmap=cmap, alpha=0.8)
                
                # Styling
                ax.set_xlim(self.visualizer.hk_bounds['lon_min'], 
                           self.visualizer.hk_bounds['lon_max'])
                ax.set_ylim(self.visualizer.hk_bounds['lat_min'], 
                           self.visualizer.hk_bounds['lat_max'])
                ax.set_aspect('equal')
                ax.set_xticks([])
                ax.set_yticks([])
                
                # Add timestamp
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                ax.set_title(f'Hong Kong Real-Time Rainfall - {timestamp}',
                            color='white', fontsize=14, pad=20)
                
                plt.draw()
                
            except Exception as e:
                print(f"Error updating display: {e}")
        
        # Set up timer for updates
        timer = fig.canvas.new_timer(interval=self.update_interval)
        timer.add_callback(update_display)
        timer.start()
        
        # Initial display
        update_display()
        plt.show()


# Main execution and examples
def main():
    """
    Main function demonstrating different visualization modes.
    """
    print("Hong Kong Rainfall Artistic Visualization")
    print("=" * 50)
    
    # Create visualizer
    visualizer = ArtisticRainfallVisualizer(style='artistic')
    
    print("1. Creating static artistic visualization...")
    static_fig = visualizer.create_static_visualization(
        save_path="/Users/zyf/Desktop/HK RAIN DATA/visualizations/rainfall_static.png"
    )
    
    print("2. Creating animated visualization...")
    try:
        anim = visualizer.create_animated_visualization(
            duration=30,
            save_path="/Users/zyf/Desktop/HK RAIN DATA/visualizations/rainfall_animation.gif"
        )
    except Exception as e:
        print(f"Animation creation failed: {e}")
    
    print("Visualization demo completed!")


if __name__ == "__main__":
    main()
