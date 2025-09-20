"""
Advanced Artistic Effects for Hong Kong Rainfall Visualization
=============================================================

This module provides additional artistic rendering effects including
generative art patterns, color-shifting landscapes, and abstract animations
based on meteorological data transformations.
"""

import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import time
import math
from typing import Dict, List, Tuple, Optional
import colorsys


class ArtisticEffects:
    """
    Advanced artistic effects for rainfall data visualization.
    """
    
    def __init__(self):
        self.time_start = time.time()
        
    def generate_organic_patterns(self, rainfall_data: np.ndarray, 
                                lon_mesh: np.ndarray, lat_mesh: np.ndarray) -> np.ndarray:
        """
        Generate organic, plant-like growth patterns based on rainfall intensity.
        
        Args:
            rainfall_data: 2D array of rainfall intensities
            lon_mesh, lat_mesh: Coordinate meshes
            
        Returns:
            Organic pattern array
        """
        # Create growth pattern based on rainfall
        time_factor = (time.time() - self.time_start) / 10
        
        # Base organic pattern
        pattern = np.zeros_like(rainfall_data)
        
        # Create multiple growth seeds based on high rainfall areas
        high_rain_points = np.where(rainfall_data > np.percentile(rainfall_data, 80))
        
        for i in range(min(len(high_rain_points[0]), 5)):  # Limit to 5 seeds
            seed_y, seed_x = high_rain_points[0][i], high_rain_points[1][i]
            seed_lon, seed_lat = lon_mesh[seed_y, seed_x], lat_mesh[seed_y, seed_x]
            
            # Calculate distance from seed
            dist = np.sqrt((lon_mesh - seed_lon)**2 + (lat_mesh - seed_lat)**2)
            
            # Create growing branches
            for angle in np.linspace(0, 2*np.pi, 8):
                branch_pattern = self._create_branch_pattern(
                    dist, angle, time_factor, rainfall_data[seed_y, seed_x]
                )
                pattern += branch_pattern
        
        return np.clip(pattern, 0, 1)
    
    def _create_branch_pattern(self, dist: np.ndarray, angle: float, 
                              time_factor: float, intensity: float) -> np.ndarray:
        """
        Create a single branch pattern.
        
        Args:
            dist: Distance array from seed point
            angle: Branch angle
            time_factor: Time-based growth factor
            intensity: Rainfall intensity at seed
            
        Returns:
            Branch pattern array
        """
        # Growth length based on time and intensity
        growth_length = (time_factor * intensity / 50) % 0.5
        
        # Create sinuous branch
        branch_width = 0.02 + intensity / 1000
        
        # Angular component for branch direction
        angular_dist = np.abs(np.arctan2(np.sin(angle), np.cos(angle)))
        
        # Branch pattern with organic variation
        branch = np.exp(-dist * 100) * np.exp(-angular_dist * 20)
        branch *= np.sin(dist * 50 + time_factor + angle) * 0.5 + 0.5
        branch *= np.exp(-dist / growth_length) if growth_length > 0 else 0
        
        return branch * intensity / 50
    
    def create_color_shifting_landscape(self, rainfall_data: np.ndarray,
                                      base_colormap: str = 'viridis') -> np.ndarray:
        """
        Create a color-shifting landscape based on rainfall patterns.
        
        Args:
            rainfall_data: 2D array of rainfall intensities
            base_colormap: Base colormap name
            
        Returns:
            RGB color array
        """
        time_factor = (time.time() - self.time_start) / 5
        
        # Create color channels based on different aspects of data
        height, width = rainfall_data.shape
        rgb_array = np.zeros((height, width, 3))
        
        # Red channel: Rainfall intensity with time modulation
        red_base = rainfall_data / (np.max(rainfall_data) + 1e-6)
        red_modulation = 0.3 * np.sin(time_factor + rainfall_data / 5)
        rgb_array[:, :, 0] = np.clip(red_base + red_modulation, 0, 1)
        
        # Green channel: Spatial gradients
        grad_y, grad_x = np.gradient(rainfall_data)
        grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        green_base = grad_magnitude / (np.max(grad_magnitude) + 1e-6)
        green_modulation = 0.2 * np.cos(time_factor * 1.5 - rainfall_data / 3)
        rgb_array[:, :, 1] = np.clip(green_base + green_modulation, 0, 1)
        
        # Blue channel: Temporal evolution
        blue_base = np.sin(rainfall_data / 10 + time_factor) * 0.5 + 0.5
        blue_modulation = 0.1 * np.sin(time_factor * 2)
        rgb_array[:, :, 2] = np.clip(blue_base + blue_modulation, 0, 1)
        
        return rgb_array
    
    def generate_abstract_particles(self, rainfall_data: np.ndarray,
                                  n_particles: int = 1000) -> Dict:
        """
        Generate abstract particle system based on rainfall data.
        
        Args:
            rainfall_data: 2D array of rainfall intensities
            n_particles: Number of particles to generate
            
        Returns:
            Dictionary with particle properties
        """
        height, width = rainfall_data.shape
        time_factor = (time.time() - self.time_start) / 3
        
        # Find high-intensity regions for particle concentration
        rain_flat = rainfall_data.flatten()
        probabilities = rain_flat / (np.sum(rain_flat) + 1e-6)
        
        # Generate particle positions based on rainfall probability
        indices = np.random.choice(len(rain_flat), size=n_particles, p=probabilities)
        y_indices, x_indices = np.unravel_index(indices, (height, width))
        
        particles = {
            'x': x_indices / width,  # Normalize to [0, 1]
            'y': y_indices / height,
            'intensity': rainfall_data[y_indices, x_indices],
            'phase': np.random.uniform(0, 2*np.pi, n_particles),
            'frequency': np.random.uniform(0.5, 3.0, n_particles),
            'size': np.random.uniform(0.5, 3.0, n_particles)
        }
        
        # Add dynamic motion
        particles['x'] += 0.1 * np.sin(time_factor * particles['frequency'] + particles['phase'])
        particles['y'] += 0.1 * np.cos(time_factor * particles['frequency'] + particles['phase'])
        
        # Keep particles in bounds
        particles['x'] = np.clip(particles['x'], 0, 1)
        particles['y'] = np.clip(particles['y'], 0, 1)
        
        return particles
    
    def create_mandala_pattern(self, rainfall_data: np.ndarray, 
                             center_intensity: float) -> np.ndarray:
        """
        Create mandala-like circular patterns based on rainfall center.
        
        Args:
            rainfall_data: 2D array of rainfall intensities
            center_intensity: Rainfall intensity at center
            
        Returns:
            Mandala pattern array
        """
        height, width = rainfall_data.shape
        time_factor = (time.time() - self.time_start) / 8
        
        # Create coordinate system centered on data
        y_center, x_center = height // 2, width // 2
        y_coords = np.arange(height) - y_center
        x_coords = np.arange(width) - x_center
        x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)
        
        # Convert to polar coordinates
        r = np.sqrt(x_mesh**2 + y_mesh**2)
        theta = np.arctan2(y_mesh, x_mesh)
        
        # Create mandala pattern
        n_petals = int(6 + center_intensity / 5)  # Number of petals based on intensity
        
        # Base circular pattern
        mandala = np.exp(-r / (10 + center_intensity))
        
        # Add petal pattern
        petal_pattern = np.sin(n_petals * theta + time_factor) * 0.5 + 0.5
        mandala *= petal_pattern
        
        # Add concentric circles
        circle_pattern = np.sin(r / 3 + time_factor) * 0.3 + 0.7
        mandala *= circle_pattern
        
        # Add time-based rotation
        rotation_offset = time_factor / n_petals
        rotated_theta = theta + rotation_offset
        rotating_pattern = np.sin(n_petals * rotated_theta) * 0.2 + 0.8
        mandala *= rotating_pattern
        
        return mandala
    
    def generate_weather_spirals(self, rainfall_data: np.ndarray,
                               flow_u: np.ndarray, flow_v: np.ndarray) -> List[Dict]:
        """
        Generate spiral patterns representing weather systems.
        
        Args:
            rainfall_data: 2D array of rainfall intensities
            flow_u, flow_v: Flow field components
            
        Returns:
            List of spiral pattern dictionaries
        """
        spirals = []
        time_factor = (time.time() - self.time_start) / 6
        
        # Find spiral centers (local maxima in rainfall with significant flow)
        flow_magnitude = np.sqrt(flow_u**2 + flow_v**2)
        combined_intensity = rainfall_data * flow_magnitude
        
        # Find local maxima
        from scipy.ndimage import maximum_filter
        try:
            local_maxima = (combined_intensity == maximum_filter(combined_intensity, size=10))
            maxima_points = np.where(local_maxima & (combined_intensity > np.percentile(combined_intensity, 90)))
            
            for i in range(min(len(maxima_points[0]), 3)):  # Limit to 3 spirals
                center_y, center_x = maxima_points[0][i], maxima_points[1][i]
                intensity = rainfall_data[center_y, center_x]
                
                spiral = self._create_spiral_pattern(
                    center_x, center_y, intensity, time_factor, rainfall_data.shape
                )
                spirals.append(spiral)
                
        except ImportError:
            # Fallback: create spirals at fixed locations
            centers = [(rainfall_data.shape[1]//3, rainfall_data.shape[0]//3),
                      (2*rainfall_data.shape[1]//3, 2*rainfall_data.shape[0]//3)]
            
            for center_x, center_y in centers:
                intensity = rainfall_data[center_y, center_x]
                spiral = self._create_spiral_pattern(
                    center_x, center_y, intensity, time_factor, rainfall_data.shape
                )
                spirals.append(spiral)
        
        return spirals
    
    def _create_spiral_pattern(self, center_x: int, center_y: int, 
                              intensity: float, time_factor: float,
                              shape: Tuple[int, int]) -> Dict:
        """
        Create a single spiral pattern.
        
        Args:
            center_x, center_y: Spiral center coordinates
            intensity: Rainfall intensity at center
            time_factor: Time-based animation factor
            shape: Array shape (height, width)
            
        Returns:
            Spiral pattern dictionary
        """
        height, width = shape
        
        # Create coordinate grids relative to center
        y_coords = np.arange(height) - center_y
        x_coords = np.arange(width) - center_x
        x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)
        
        # Convert to polar coordinates
        r = np.sqrt(x_mesh**2 + y_mesh**2)
        theta = np.arctan2(y_mesh, x_mesh)
        
        # Create spiral
        spiral_tightness = 0.3 + intensity / 100
        spiral_angle = theta + spiral_tightness * r + time_factor
        
        # Spiral pattern
        spiral_pattern = np.sin(spiral_angle) * np.exp(-r / (20 + intensity))
        spiral_pattern = np.maximum(spiral_pattern, 0)
        
        return {
            'pattern': spiral_pattern,
            'center': (center_x, center_y),
            'intensity': intensity,
            'rotation': time_factor
        }
    
    def apply_chromatic_aberration(self, image: np.ndarray, 
                                 intensity: float = 0.02) -> np.ndarray:
        """
        Apply chromatic aberration effect based on rainfall intensity.
        
        Args:
            image: RGB image array
            intensity: Aberration intensity
            
        Returns:
            Image with chromatic aberration
        """
        if len(image.shape) != 3 or image.shape[2] != 3:
            return image
        
        height, width = image.shape[:2]
        center_x, center_y = width // 2, height // 2
        
        # Create coordinate grids
        y_coords = np.arange(height) - center_y
        x_coords = np.arange(width) - center_x
        x_mesh, y_mesh = np.meshgrid(x_coords, y_coords)
        
        # Distance from center
        r = np.sqrt(x_mesh**2 + y_mesh**2)
        
        # Apply different shifts to RGB channels
        aberrated_image = np.zeros_like(image)
        
        # Red channel - shift outward
        shift_r = intensity * r / max(width, height)
        x_shifted_r = x_mesh + shift_r * x_mesh / (r + 1e-6)
        y_shifted_r = y_mesh + shift_r * y_mesh / (r + 1e-6)
        
        # Green channel - no shift
        aberrated_image[:, :, 1] = image[:, :, 1]
        
        # Blue channel - shift inward
        shift_b = -intensity * r / max(width, height)
        x_shifted_b = x_mesh + shift_b * x_mesh / (r + 1e-6)
        y_shifted_b = y_mesh + shift_b * y_mesh / (r + 1e-6)
        
        # Apply shifts with bounds checking
        for y in range(height):
            for x in range(width):
                # Red channel
                new_x_r = int(x + x_shifted_r[y, x])
                new_y_r = int(y + y_shifted_r[y, x])
                if 0 <= new_x_r < width and 0 <= new_y_r < height:
                    aberrated_image[new_y_r, new_x_r, 0] = image[y, x, 0]
                
                # Blue channel
                new_x_b = int(x + x_shifted_b[y, x])
                new_y_b = int(y + y_shifted_b[y, x])
                if 0 <= new_x_b < width and 0 <= new_y_b < height:
                    aberrated_image[new_y_b, new_x_b, 2] = image[y, x, 2]
        
        return aberrated_image
    
    def create_data_driven_music_visualization(self, rainfall_data: np.ndarray) -> Dict:
        """
        Create visual patterns that could represent rainfall as music.
        
        Args:
            rainfall_data: 2D array of rainfall intensities
            
        Returns:
            Dictionary with musical visualization elements
        """
        time_factor = (time.time() - self.time_start) / 4
        
        # Convert rainfall to frequency domain representation
        rainfall_1d = rainfall_data.flatten()
        
        # Create frequency bins (simulating audio frequencies)
        n_frequencies = 32
        frequency_bins = np.zeros(n_frequencies)
        
        # Map rainfall intensities to frequency bins
        for i, rain_val in enumerate(rainfall_1d):
            if rain_val > 0:
                freq_bin = int((rain_val / np.max(rainfall_data)) * (n_frequencies - 1))
                frequency_bins[freq_bin] += rain_val
        
        # Normalize
        frequency_bins = frequency_bins / (np.max(frequency_bins) + 1e-6)
        
        # Create visual waveform
        waveform_x = np.linspace(0, 2*np.pi, len(rainfall_1d))
        waveform_y = rainfall_1d * np.sin(waveform_x + time_factor)
        
        # Create spectral visualization
        spectral_pattern = np.zeros((n_frequencies, 50))
        for i, amp in enumerate(frequency_bins):
            spectral_pattern[i, :] = amp * np.sin(np.linspace(0, 4*np.pi, 50) + time_factor)
        
        return {
            'frequency_bins': frequency_bins,
            'waveform_x': waveform_x,
            'waveform_y': waveform_y,
            'spectral_pattern': spectral_pattern,
            'beat_intensity': np.mean(rainfall_data)
        }


class GenerativeArtRenderer:
    """
    Renders generative art based on weather patterns.
    """
    
    def __init__(self):
        self.effects = ArtisticEffects()
        
    def render_organic_growth(self, rainfall_data: np.ndarray, 
                            lon_mesh: np.ndarray, lat_mesh: np.ndarray,
                            ax: plt.Axes) -> None:
        """
        Render organic growth patterns on the given axes.
        
        Args:
            rainfall_data: 2D array of rainfall intensities
            lon_mesh, lat_mesh: Coordinate meshes
            ax: Matplotlib axes to render on
        """
        # Generate organic pattern
        organic_pattern = self.effects.generate_organic_patterns(
            rainfall_data, lon_mesh, lat_mesh
        )
        
        # Render as contours with organic colors
        levels = np.linspace(0, np.max(organic_pattern), 10)
        colors = ['#003300', '#006600', '#009900', '#00cc00', '#00ff00']
        
        ax.contourf(lon_mesh, lat_mesh, organic_pattern, 
                   levels=levels, colors=colors, alpha=0.6)
        
        # Add glowing effect
        glow_pattern = organic_pattern * 0.3
        ax.contour(lon_mesh, lat_mesh, glow_pattern, 
                  levels=levels[::2], colors='white', alpha=0.3, linewidths=1)
    
    def render_abstract_mandala(self, rainfall_data: np.ndarray,
                              lon_mesh: np.ndarray, lat_mesh: np.ndarray,
                              ax: plt.Axes) -> None:
        """
        Render mandala patterns based on rainfall centers.
        
        Args:
            rainfall_data: 2D array of rainfall intensities
            lon_mesh, lat_mesh: Coordinate meshes
            ax: Matplotlib axes to render on
        """
        # Find rainfall center
        center_intensity = np.max(rainfall_data)
        
        # Generate mandala
        mandala_pattern = self.effects.create_mandala_pattern(
            rainfall_data, center_intensity
        )
        
        # Create custom colormap for mandala
        colors = ['#000033', '#003366', '#006699', '#0099cc', '#00ccff']
        from matplotlib.colors import LinearSegmentedColormap
        mandala_cmap = LinearSegmentedColormap.from_list('mandala', colors)
        
        # Render mandala
        ax.imshow(mandala_pattern, extent=[
            np.min(lon_mesh), np.max(lon_mesh),
            np.min(lat_mesh), np.max(lat_mesh)
        ], cmap=mandala_cmap, alpha=0.7, origin='lower')
    
    def render_weather_spirals(self, rainfall_data: np.ndarray,
                             flow_u: np.ndarray, flow_v: np.ndarray,
                             lon_mesh: np.ndarray, lat_mesh: np.ndarray,
                             ax: plt.Axes) -> None:
        """
        Render spiral weather patterns.
        
        Args:
            rainfall_data: 2D array of rainfall intensities
            flow_u, flow_v: Flow field components
            lon_mesh, lat_mesh: Coordinate meshes
            ax: Matplotlib axes to render on
        """
        spirals = self.effects.generate_weather_spirals(rainfall_data, flow_u, flow_v)
        
        for spiral in spirals:
            pattern = spiral['pattern']
            intensity = spiral['intensity']
            
            # Color based on intensity
            color_intensity = intensity / np.max(rainfall_data) if np.max(rainfall_data) > 0 else 0.5
            colors = plt.cm.plasma(np.linspace(0, color_intensity, 10))
            
            # Render spiral
            levels = np.linspace(0, np.max(pattern), 8)
            ax.contour(lon_mesh, lat_mesh, pattern, 
                      levels=levels, colors=['white'], alpha=0.4, linewidths=1.5)


# Example usage demonstration
def demonstrate_artistic_effects():
    """
    Demonstrate various artistic effects with sample data.
    """
    print("Demonstrating Artistic Effects for Rainfall Visualization")
    print("=" * 60)
    
    # Create sample data
    x = np.linspace(0, 10, 50)
    y = np.linspace(0, 8, 40)
    X, Y = np.meshgrid(x, y)
    
    # Sample rainfall pattern
    rainfall = (20 * np.exp(-((X-5)**2 + (Y-4)**2) / 4) + 
               10 * np.exp(-((X-3)**2 + (Y-6)**2) / 2) +
               5 * np.random.random(X.shape))
    
    # Sample flow field
    flow_u = -np.gradient(rainfall, axis=1)
    flow_v = -np.gradient(rainfall, axis=0)
    
    # Create effects
    effects = ArtisticEffects()
    renderer = GenerativeArtRenderer()
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='black')
    
    for ax in axes.flat:
        ax.set_facecolor('black')
    
    # Test 1: Organic patterns
    organic = effects.generate_organic_patterns(rainfall, X, Y)
    axes[0,0].contourf(X, Y, organic, levels=20, cmap='Greens', alpha=0.8)
    axes[0,0].set_title('Organic Growth Patterns', color='white')
    
    # Test 2: Color-shifting landscape
    color_landscape = effects.create_color_shifting_landscape(rainfall)
    axes[0,1].imshow(color_landscape, extent=[0, 10, 0, 8], origin='lower')
    axes[0,1].set_title('Color-Shifting Landscape', color='white')
    
    # Test 3: Mandala pattern
    mandala = effects.create_mandala_pattern(rainfall, np.max(rainfall))
    axes[1,0].imshow(mandala, extent=[0, 10, 0, 8], cmap='plasma', origin='lower')
    axes[1,0].set_title('Rainfall Mandala', color='white')
    
    # Test 4: Combined artistic effects
    axes[1,1].contourf(X, Y, rainfall, levels=15, cmap='viridis', alpha=0.6)
    renderer.render_weather_spirals(rainfall, flow_u, flow_v, X, Y, axes[1,1])
    axes[1,1].set_title('Weather Spirals', color='white')
    
    # Style all axes
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig('/Users/zyf/Desktop/HK RAIN DATA/visualizations/artistic_effects_demo.png',
                dpi=300, facecolor='black', bbox_inches='tight')
    plt.show()
    
    print("Artistic effects demonstration completed!")


if __name__ == "__main__":
    demonstrate_artistic_effects()
