"""
Data Processing Pipeline for Hong Kong Rainfall Data
====================================================

This module provides data cleaning, transformation, and preparation functions
for creating artistic rainfall visualizations.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import json
from scipy import interpolate
from scipy.spatial.distance import cdist
import warnings
warnings.filterwarnings('ignore')


class RainfallDataProcessor:
    """
    A class to process and transform rainfall data for artistic visualization.
    """
    
    def __init__(self):
        # Hong Kong geographical bounds
        self.hk_bounds = {
            'lat_min': 22.15,
            'lat_max': 22.58,
            'lon_min': 113.83,
            'lon_max': 114.41
        }
        
        # Grid resolution for interpolation
        self.grid_resolution = 50
        
    def clean_raw_data(self, raw_data: Dict) -> pd.DataFrame:
        """
        Clean and standardize raw rainfall data.
        
        Args:
            raw_data: Raw data dictionary from data fetcher
            
        Returns:
            Cleaned DataFrame
        """
        records = []
        
        for district, info in raw_data.items():
            record = {
                'district': district,
                'district_en': info.get('name_en', ''),
                'rainfall_mm': float(info.get('rainfall_mm', 0)),
                'latitude': float(info.get('lat', 0)),
                'longitude': float(info.get('lon', 0)),
                'timestamp': pd.to_datetime(info.get('timestamp', datetime.now()))
            }
            records.append(record)
        
        df = pd.DataFrame(records)
        
        # Remove invalid data points
        df = df[df['rainfall_mm'] >= 0]
        df = df[df['latitude'] > 0]
        df = df[df['longitude'] > 0]
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        return df
    
    def interpolate_rainfall_grid(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Interpolate rainfall data onto a regular grid for smooth visualization.
        
        Args:
            df: DataFrame with rainfall data
            
        Returns:
            Tuple of (longitude_grid, latitude_grid, rainfall_grid)
        """
        # Create regular grid
        lon_grid = np.linspace(self.hk_bounds['lon_min'], self.hk_bounds['lon_max'], self.grid_resolution)
        lat_grid = np.linspace(self.hk_bounds['lat_min'], self.hk_bounds['lat_max'], self.grid_resolution)
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        
        # Prepare data points
        points = df[['longitude', 'latitude']].values
        values = df['rainfall_mm'].values
        
        # Add boundary points with zero rainfall to prevent edge effects
        boundary_points = self._create_boundary_points()
        points = np.vstack([points, boundary_points])
        values = np.hstack([values, np.zeros(len(boundary_points))])
        
        # Interpolate using radial basis function
        try:
            from scipy.interpolate import Rbf
            rbf = Rbf(points[:, 0], points[:, 1], values, function='multiquadric', smooth=0.1)
            rainfall_grid = rbf(lon_mesh, lat_mesh)
        except:
            # Fallback to griddata interpolation
            from scipy.interpolate import griddata
            grid_points = np.column_stack([lon_mesh.ravel(), lat_mesh.ravel()])
            rainfall_interpolated = griddata(points, values, grid_points, method='cubic', fill_value=0)
            rainfall_grid = rainfall_interpolated.reshape(lon_mesh.shape)
        
        # Ensure non-negative values
        rainfall_grid = np.maximum(rainfall_grid, 0)
        
        return lon_mesh, lat_mesh, rainfall_grid
    
    def _create_boundary_points(self) -> np.ndarray:
        """
        Create boundary points around Hong Kong for interpolation.
        
        Returns:
            Array of boundary coordinates
        """
        boundary_points = []
        
        # Create points around the boundary
        n_boundary = 20
        
        # Top and bottom boundaries
        for lon in np.linspace(self.hk_bounds['lon_min'], self.hk_bounds['lon_max'], n_boundary):
            boundary_points.append([lon, self.hk_bounds['lat_min']])
            boundary_points.append([lon, self.hk_bounds['lat_max']])
        
        # Left and right boundaries
        for lat in np.linspace(self.hk_bounds['lat_min'], self.hk_bounds['lat_max'], n_boundary):
            boundary_points.append([self.hk_bounds['lon_min'], lat])
            boundary_points.append([self.hk_bounds['lon_max'], lat])
        
        return np.array(boundary_points)
    
    def calculate_rainfall_gradients(self, lon_mesh: np.ndarray, lat_mesh: np.ndarray, 
                                   rainfall_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate rainfall gradients for flow visualization.
        
        Args:
            lon_mesh: Longitude mesh grid
            lat_mesh: Latitude mesh grid  
            rainfall_grid: Rainfall intensity grid
            
        Returns:
            Tuple of (gradient_x, gradient_y)
        """
        # Calculate gradients
        grad_y, grad_x = np.gradient(rainfall_grid)
        
        # Normalize gradients by grid spacing
        dlon = lon_mesh[0, 1] - lon_mesh[0, 0]
        dlat = lat_mesh[1, 0] - lat_mesh[0, 0]
        
        grad_x = grad_x / dlon
        grad_y = grad_y / dlat
        
        return grad_x, grad_y
    
    def generate_flow_field(self, lon_mesh: np.ndarray, lat_mesh: np.ndarray, 
                           rainfall_grid: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate flow field for artistic water flow visualization.
        
        Args:
            lon_mesh: Longitude mesh grid
            lat_mesh: Latitude mesh grid
            rainfall_grid: Rainfall intensity grid
            
        Returns:
            Tuple of (flow_u, flow_v) velocity components
        """
        # Calculate gradients
        grad_x, grad_y = self.calculate_rainfall_gradients(lon_mesh, lat_mesh, rainfall_grid)
        
        # Create flow field based on rainfall intensity and gradients
        flow_u = -grad_x + 0.1 * np.random.randn(*grad_x.shape)  # Flow towards low pressure
        flow_v = -grad_y + 0.1 * np.random.randn(*grad_y.shape)
        
        # Add swirling patterns based on rainfall intensity
        time_factor = datetime.now().timestamp() / 1000
        swirl_u = 0.5 * rainfall_grid * np.sin(time_factor + lon_mesh)
        swirl_v = 0.5 * rainfall_grid * np.cos(time_factor + lat_mesh)
        
        flow_u += swirl_u
        flow_v += swirl_v
        
        # Normalize flow field
        flow_magnitude = np.sqrt(flow_u**2 + flow_v**2)
        flow_magnitude[flow_magnitude == 0] = 1  # Avoid division by zero
        
        flow_u = flow_u / flow_magnitude * np.sqrt(rainfall_grid + 0.1)
        flow_v = flow_v / flow_magnitude * np.sqrt(rainfall_grid + 0.1)
        
        return flow_u, flow_v
    
    def temporal_smoothing(self, historical_data: List[pd.DataFrame], 
                          smoothing_window: int = 5) -> pd.DataFrame:
        """
        Apply temporal smoothing to reduce noise in time series data.
        
        Args:
            historical_data: List of DataFrames with historical rainfall data
            smoothing_window: Window size for rolling average
            
        Returns:
            Smoothed DataFrame
        """
        if not historical_data:
            return pd.DataFrame()
        
        # Combine all data
        combined_df = pd.concat(historical_data, ignore_index=True)
        
        # Sort by district and timestamp
        combined_df = combined_df.sort_values(['district', 'timestamp'])
        
        # Apply rolling average by district
        smoothed_data = []
        
        for district in combined_df['district'].unique():
            district_data = combined_df[combined_df['district'] == district].copy()
            district_data['rainfall_mm_smooth'] = district_data['rainfall_mm'].rolling(
                window=smoothing_window, center=True, min_periods=1
            ).mean()
            smoothed_data.append(district_data)
        
        return pd.concat(smoothed_data, ignore_index=True)
    
    def create_rainfall_clusters(self, df: pd.DataFrame, n_clusters: int = 5) -> pd.DataFrame:
        """
        Create rainfall intensity clusters for color mapping.
        
        Args:
            df: DataFrame with rainfall data
            n_clusters: Number of intensity clusters
            
        Returns:
            DataFrame with cluster assignments
        """
        try:
            from sklearn.cluster import KMeans
            
            # Prepare features: rainfall intensity and location
            features = df[['rainfall_mm', 'latitude', 'longitude']].values
            
            # Normalize features
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            
            # Apply clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            df['rainfall_cluster'] = kmeans.fit_predict(features_scaled)
            
        except ImportError:
            # Fallback to simple intensity-based clustering
            rainfall_percentiles = np.percentile(df['rainfall_mm'], 
                                               np.linspace(0, 100, n_clusters + 1))
            df['rainfall_cluster'] = pd.cut(df['rainfall_mm'], 
                                          bins=rainfall_percentiles, 
                                          labels=False, 
                                          include_lowest=True)
            df['rainfall_cluster'] = df['rainfall_cluster'].fillna(0)
        
        return df
    
    def calculate_weather_patterns(self, df: pd.DataFrame) -> Dict:
        """
        Calculate weather pattern metrics for artistic visualization.
        
        Args:
            df: DataFrame with rainfall data
            
        Returns:
            Dictionary with pattern metrics
        """
        patterns = {}
        
        # Overall intensity
        patterns['total_rainfall'] = df['rainfall_mm'].sum()
        patterns['max_rainfall'] = df['rainfall_mm'].max()
        patterns['mean_rainfall'] = df['rainfall_mm'].mean()
        patterns['std_rainfall'] = df['rainfall_mm'].std()
        
        # Spatial distribution
        patterns['rainfall_range_lat'] = df['latitude'].max() - df['latitude'].min()
        patterns['rainfall_range_lon'] = df['longitude'].max() - df['longitude'].min()
        
        # Calculate center of mass
        if patterns['total_rainfall'] > 0:
            patterns['center_lat'] = np.average(df['latitude'], weights=df['rainfall_mm'])
            patterns['center_lon'] = np.average(df['longitude'], weights=df['rainfall_mm'])
        else:
            patterns['center_lat'] = df['latitude'].mean()
            patterns['center_lon'] = df['longitude'].mean()
        
        # Concentration index (how concentrated the rainfall is)
        if len(df) > 1:
            distances = cdist(df[['latitude', 'longitude']], 
                            [[patterns['center_lat'], patterns['center_lon']]])
            patterns['concentration'] = 1 / (1 + np.average(distances.flatten(), 
                                                           weights=df['rainfall_mm']))
        else:
            patterns['concentration'] = 1.0
        
        # Storm intensity classification
        if patterns['max_rainfall'] > 30:
            patterns['storm_class'] = 'heavy'
        elif patterns['max_rainfall'] > 10:
            patterns['storm_class'] = 'moderate'
        elif patterns['max_rainfall'] > 1:
            patterns['storm_class'] = 'light'
        else:
            patterns['storm_class'] = 'none'
        
        return patterns
    
    def prepare_animation_data(self, historical_df: pd.DataFrame, 
                             time_steps: int = 60) -> List[Dict]:
        """
        Prepare data for time-series animation.
        
        Args:
            historical_df: DataFrame with historical rainfall data
            time_steps: Number of time steps for animation
            
        Returns:
            List of dictionaries with animation frame data
        """
        # Get unique timestamps
        timestamps = sorted(historical_df['timestamp'].unique())
        
        # Select evenly spaced timestamps for animation
        if len(timestamps) > time_steps:
            step_size = len(timestamps) // time_steps
            selected_timestamps = timestamps[::step_size][:time_steps]
        else:
            selected_timestamps = timestamps
        
        animation_frames = []
        
        for timestamp in selected_timestamps:
            frame_data = historical_df[historical_df['timestamp'] == timestamp].copy()
            
            if not frame_data.empty:
                # Process frame data
                lon_mesh, lat_mesh, rainfall_grid = self.interpolate_rainfall_grid(frame_data)
                flow_u, flow_v = self.generate_flow_field(lon_mesh, lat_mesh, rainfall_grid)
                patterns = self.calculate_weather_patterns(frame_data)
                
                animation_frames.append({
                    'timestamp': timestamp,
                    'data': frame_data,
                    'lon_mesh': lon_mesh,
                    'lat_mesh': lat_mesh,
                    'rainfall_grid': rainfall_grid,
                    'flow_u': flow_u,
                    'flow_v': flow_v,
                    'patterns': patterns
                })
        
        return animation_frames


class DataExporter:
    """
    Class for exporting processed data to different formats.
    """
    
    @staticmethod
    def export_to_json(data: Dict, filepath: str):
        """Export data to JSON format."""
        # Convert numpy arrays to lists for JSON serialization
        exportable_data = DataExporter._make_json_serializable(data)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(exportable_data, f, ensure_ascii=False, indent=2, default=str)
    
    @staticmethod
    def export_to_csv(df: pd.DataFrame, filepath: str):
        """Export DataFrame to CSV format."""
        df.to_csv(filepath, index=False, encoding='utf-8')
    
    @staticmethod
    def _make_json_serializable(obj):
        """Convert numpy arrays and other non-serializable objects to JSON-compatible format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: DataExporter._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [DataExporter._make_json_serializable(item) for item in obj]
        else:
            return obj


# Example usage and testing
if __name__ == "__main__":
    print("Testing Rainfall Data Processor")
    print("=" * 40)
    
    processor = RainfallDataProcessor()
    
    # Create sample data for testing
    sample_data = {
        '中西區': {'rainfall_mm': 5.2, 'lat': 22.2783, 'lon': 114.1747, 'name_en': 'Central and Western', 'timestamp': datetime.now()},
        '南區': {'rainfall_mm': 3.1, 'lat': 22.2461, 'lon': 114.1661, 'name_en': 'Southern', 'timestamp': datetime.now()},
        '東區': {'rainfall_mm': 7.8, 'lat': 22.2741, 'lon': 114.2302, 'name_en': 'Eastern', 'timestamp': datetime.now()},
        '觀塘': {'rainfall_mm': 12.5, 'lat': 22.3092, 'lon': 114.2249, 'name_en': 'Kwun Tong', 'timestamp': datetime.now()}
    }
    
    # Test data cleaning
    df = processor.clean_raw_data(sample_data)
    print(f"Cleaned data shape: {df.shape}")
    print(df.head())
    
    # Test interpolation
    lon_mesh, lat_mesh, rainfall_grid = processor.interpolate_rainfall_grid(df)
    print(f"Grid shape: {rainfall_grid.shape}")
    print(f"Rainfall range: {rainfall_grid.min():.2f} - {rainfall_grid.max():.2f} mm")
    
    # Test flow field generation
    flow_u, flow_v = processor.generate_flow_field(lon_mesh, lat_mesh, rainfall_grid)
    print(f"Flow field range U: {flow_u.min():.2f} - {flow_u.max():.2f}")
    print(f"Flow field range V: {flow_v.min():.2f} - {flow_v.max():.2f}")
    
    # Test pattern calculation
    patterns = processor.calculate_weather_patterns(df)
    print("Weather patterns:")
    for key, value in patterns.items():
        print(f"  {key}: {value}")
    
    print("Data processing pipeline test completed successfully!")
