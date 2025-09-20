"""
Hong Kong Observatory Rainfall Data Fetcher
===========================================

This module provides functionality to fetch real-time rainfall data from the
Hong Kong Observatory website and simulate additional data for comprehensive
visualization purposes.
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import re
import json
from typing import Dict, List, Tuple, Optional


class HKORainfallDataFetcher:
    """
    A class to fetch and simulate Hong Kong rainfall data for artistic visualization.
    """
    
    def __init__(self):
        self.base_url = "https://www.hko.gov.hk"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Hong Kong districts with approximate coordinates
        self.hk_districts = {
            '中西區': {'lat': 22.2783, 'lon': 114.1747, 'name_en': 'Central and Western'},
            '南區': {'lat': 22.2461, 'lon': 114.1661, 'name_en': 'Southern'},
            '東區': {'lat': 22.2741, 'lon': 114.2302, 'name_en': 'Eastern'},
            '灣仔': {'lat': 22.2793, 'lon': 114.1747, 'name_en': 'Wan Chai'},
            '九龍城': {'lat': 22.3193, 'lon': 114.1914, 'name_en': 'Kowloon City'},
            '觀塘': {'lat': 22.3092, 'lon': 114.2249, 'name_en': 'Kwun Tong'},
            '深水埗': {'lat': 22.3307, 'lon': 114.1628, 'name_en': 'Sham Shui Po'},
            '黃大仙': {'lat': 22.3429, 'lon': 114.1934, 'name_en': 'Wong Tai Sin'},
            '油尖旺': {'lat': 22.3069, 'lon': 114.1722, 'name_en': 'Yau Tsim Mong'},
            '葵青': {'lat': 22.3645, 'lon': 114.1311, 'name_en': 'Kwai Tsing'},
            '荃灣': {'lat': 22.3747, 'lon': 114.1161, 'name_en': 'Tsuen Wan'},
            '屯門': {'lat': 22.3913, 'lon': 113.9759, 'name_en': 'Tuen Mun'},
            '元朗': {'lat': 22.4484, 'lon': 114.0346, 'name_en': 'Yuen Long'},
            '北區': {'lat': 22.4964, 'lon': 114.1486, 'name_en': 'North'},
            '大埔': {'lat': 22.4446, 'lon': 114.1747, 'name_en': 'Tai Po'},
            '沙田': {'lat': 22.3818, 'lon': 114.1877, 'name_en': 'Sha Tin'},
            '西貢': {'lat': 22.3815, 'lon': 114.2674, 'name_en': 'Sai Kung'},
            '離島區': {'lat': 22.2587, 'lon': 113.9422, 'name_en': 'Islands'}
        }
        
    def fetch_current_rainfall_data(self) -> Dict:
        """
        Fetch current rainfall data from HKO website.
        
        Returns:
            Dict containing rainfall data by district
        """
        try:
            # Try to fetch real data from HKO
            url = f"{self.base_url}/textonly/current/rainfall_sr_uc.htm"
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            rainfall_data = self._parse_rainfall_data(soup)
            
            if not rainfall_data:
                # Fallback to simulated data if parsing fails
                rainfall_data = self._generate_simulated_data()
                
        except Exception as e:
            print(f"Error fetching real data: {e}. Using simulated data.")
            rainfall_data = self._generate_simulated_data()
            
        return rainfall_data
    
    def _parse_rainfall_data(self, soup: BeautifulSoup) -> Dict:
        """
        Parse rainfall data from HKO webpage HTML.
        
        Args:
            soup: BeautifulSoup object of the webpage
            
        Returns:
            Dict containing parsed rainfall data
        """
        rainfall_data = {}
        current_time = datetime.now()
        
        # Look for rainfall data in the HTML
        text_content = soup.get_text()
        
        # Pattern to match rainfall data like "中 西 區1 毫 米"
        rainfall_pattern = r'([一-龯]+\s*[一-龯]*\s*區)\s*(\d+(?:\s*至\s*\d+)?)\s*毫\s*米'
        matches = re.findall(rainfall_pattern, text_content)
        
        for district, rainfall in matches:
            # Clean district name
            district_clean = district.replace(' ', '').replace('區', '區')
            
            # Parse rainfall value (handle ranges like "0 至 5")
            rainfall_clean = rainfall.replace(' ', '')
            if '至' in rainfall_clean:
                # Take the maximum value from range
                values = rainfall_clean.split('至')
                rainfall_value = float(values[-1])
            else:
                rainfall_value = float(rainfall_clean)
                
            if district_clean in self.hk_districts:
                rainfall_data[district_clean] = {
                    'rainfall_mm': rainfall_value,
                    'timestamp': current_time,
                    'lat': self.hk_districts[district_clean]['lat'],
                    'lon': self.hk_districts[district_clean]['lon'],
                    'name_en': self.hk_districts[district_clean]['name_en']
                }
        
        return rainfall_data
    
    def _generate_simulated_data(self) -> Dict:
        """
        Generate simulated rainfall data for visualization purposes.
        
        Returns:
            Dict containing simulated rainfall data
        """
        np.random.seed(int(time.time()) % 1000)  # Semi-random seed
        current_time = datetime.now()
        
        # Create weather patterns - simulate a storm system moving across HK
        storm_center_lat = 22.3 + 0.1 * np.sin(time.time() / 3600)
        storm_center_lon = 114.15 + 0.1 * np.cos(time.time() / 3600)
        
        rainfall_data = {}
        
        for district, info in self.hk_districts.items():
            # Calculate distance from storm center
            lat_diff = info['lat'] - storm_center_lat
            lon_diff = info['lon'] - storm_center_lon
            distance = np.sqrt(lat_diff**2 + lon_diff**2)
            
            # Generate rainfall based on distance from storm center
            base_rainfall = max(0, 15 - distance * 50)  # Stronger rain near center
            
            # Add some randomness and temporal variation
            time_factor = np.sin(time.time() / 1800) * 5  # 30-minute cycle
            random_factor = np.random.exponential(3)
            
            rainfall_mm = max(0, base_rainfall + time_factor + random_factor)
            
            rainfall_data[district] = {
                'rainfall_mm': round(rainfall_mm, 1),
                'timestamp': current_time,
                'lat': info['lat'],
                'lon': info['lon'],
                'name_en': info['name_en']
            }
            
        return rainfall_data
    
    def get_historical_simulation(self, hours: int = 24) -> pd.DataFrame:
        """
        Generate simulated historical rainfall data for the past N hours.
        
        Args:
            hours: Number of hours of historical data to generate
            
        Returns:
            DataFrame with historical rainfall data
        """
        historical_data = []
        current_time = datetime.now()
        
        for i in range(hours * 6):  # Every 10 minutes
            timestamp = current_time - timedelta(minutes=i * 10)
            
            # Simulate different weather patterns over time
            storm_intensity = np.sin(i / 50) + 1  # Varying storm intensity
            
            for district, info in self.hk_districts.items():
                # Simulate rainfall with patterns
                base_rain = np.random.exponential(2) * storm_intensity
                
                # Add geographical patterns (more rain in mountains)
                if district in ['大埔', '北區', '西貢']:
                    base_rain *= 1.3
                elif district in ['中西區', '油尖旺']:
                    base_rain *= 0.7
                
                historical_data.append({
                    'timestamp': timestamp,
                    'district': district,
                    'district_en': info['name_en'],
                    'rainfall_mm': max(0, round(base_rain, 1)),
                    'lat': info['lat'],
                    'lon': info['lon']
                })
        
        return pd.DataFrame(historical_data).sort_values('timestamp')
    
    def save_data(self, data: Dict, filename: str = None):
        """
        Save rainfall data to JSON file.
        
        Args:
            data: Rainfall data dictionary
            filename: Optional filename, defaults to timestamp
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rainfall_data_{timestamp}.json"
        
        # Convert datetime objects to strings for JSON serialization
        serializable_data = {}
        for district, info in data.items():
            serializable_data[district] = {
                'rainfall_mm': info['rainfall_mm'],
                'timestamp': info['timestamp'].isoformat(),
                'lat': info['lat'],
                'lon': info['lon'],
                'name_en': info['name_en']
            }
        
        filepath = f"/Users/zyf/Desktop/HK RAIN DATA/data/{filename}"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_data, f, ensure_ascii=False, indent=2)
        
        print(f"Data saved to {filepath}")


class RealTimeDataStream:
    """
    A class to provide real-time streaming of rainfall data for live visualization.
    """
    
    def __init__(self, update_interval: int = 300):  # 5 minutes default
        self.fetcher = HKORainfallDataFetcher()
        self.update_interval = update_interval
        self.is_running = False
        
    def start_stream(self):
        """
        Start the real-time data stream.
        """
        self.is_running = True
        print("Starting real-time rainfall data stream...")
        
        while self.is_running:
            try:
                data = self.fetcher.fetch_current_rainfall_data()
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                print(f"\n[{timestamp}] Latest Rainfall Data:")
                for district, info in data.items():
                    print(f"  {district}: {info['rainfall_mm']} mm")
                
                # Save current data
                self.fetcher.save_data(data)
                
                # Wait for next update
                time.sleep(self.update_interval)
                
            except KeyboardInterrupt:
                print("\nStopping data stream...")
                self.is_running = False
            except Exception as e:
                print(f"Error in data stream: {e}")
                time.sleep(60)  # Wait 1 minute before retry
    
    def stop_stream(self):
        """
        Stop the real-time data stream.
        """
        self.is_running = False


# Example usage and testing
if __name__ == "__main__":
    # Test the data fetcher
    fetcher = HKORainfallDataFetcher()
    
    print("Testing Hong Kong Rainfall Data Fetcher")
    print("=" * 50)
    
    # Fetch current data
    current_data = fetcher.fetch_current_rainfall_data()
    print(f"Fetched data for {len(current_data)} districts:")
    
    for district, info in current_data.items():
        print(f"  {district} ({info['name_en']}): {info['rainfall_mm']} mm")
    
    # Save current data
    fetcher.save_data(current_data)
    
    # Generate historical data
    print("\nGenerating historical data...")
    historical_df = fetcher.get_historical_simulation(hours=6)
    print(f"Generated {len(historical_df)} historical records")
    
    # Save historical data
    historical_filepath = "/Users/zyf/Desktop/HK RAIN DATA/data/historical_rainfall.csv"
    historical_df.to_csv(historical_filepath, index=False, encoding='utf-8')
    print(f"Historical data saved to {historical_filepath}")
