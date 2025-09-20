#!/usr/bin/env python3
"""
Hong Kong Rainfall Data Artistic Visualization
==============================================

Main execution script for creating artistic rainfall visualizations
using Hong Kong Observatory data.

This script demonstrates the transformation of natural meteorological data
into engaging artistic expressions that explore the boundary between
data visualization and artistic expression.

Author: Data Visualization Artist
Date: September 2025
"""

import sys
import os
import argparse
from datetime import datetime
import time

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import project modules
try:
    from data_fetcher import HKORainfallDataFetcher, RealTimeDataStream
    from data_processor import RainfallDataProcessor
    from rainfall_visualizer import ArtisticRainfallVisualizer, RealTimeVisualizer
    from artistic_effects import ArtisticEffects, GenerativeArtRenderer
    print("✓ All modules imported successfully")
except ImportError as e:
    print(f"⚠ Warning: Some modules could not be imported: {e}")
    print("  This is expected if dependencies are not yet installed.")
    print("  Please run: pip install -r requirements.txt")


class HKRainfallArtProject:
    """
    Main project class for Hong Kong Rainfall Artistic Visualization.
    """
    
    def __init__(self):
        self.project_root = os.path.dirname(__file__)
        self.data_dir = os.path.join(self.project_root, 'data')
        self.viz_dir = os.path.join(self.project_root, 'visualizations')
        self.assets_dir = os.path.join(self.project_root, 'assets')
        
        # Create directories if they don't exist
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.viz_dir, exist_ok=True)
        os.makedirs(self.assets_dir, exist_ok=True)
        
        print(f"🎨 Hong Kong Rainfall Art Project initialized")
        print(f"   Project root: {self.project_root}")
        print(f"   Data directory: {self.data_dir}")
        print(f"   Visualization directory: {self.viz_dir}")
    
    def check_dependencies(self) -> bool:
        """
        Check if all required dependencies are available.
        
        Returns:
            True if all dependencies are available, False otherwise
        """
        required_packages = [
            'matplotlib', 'numpy', 'pandas', 'requests', 
            'beautifulsoup4', 'scipy', 'pillow'
        ]
        
        missing_packages = []
        
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ Missing packages: {', '.join(missing_packages)}")
            print("   Please install with: pip install -r requirements.txt")
            return False
        else:
            print("✓ All dependencies available")
            return True
    
    def demonstrate_data_fetching(self):
        """
        Demonstrate data fetching capabilities.
        """
        print("\n📊 Demonstrating Data Fetching")
        print("-" * 40)
        
        try:
            # Initialize data fetcher
            fetcher = HKORainfallDataFetcher()
            
            # Fetch current data
            print("Fetching current Hong Kong rainfall data...")
            current_data = fetcher.fetch_current_rainfall_data()
            
            print(f"✓ Fetched data for {len(current_data)} districts:")
            for district, info in list(current_data.items())[:5]:  # Show first 5
                print(f"   {district}: {info['rainfall_mm']} mm")
            
            if len(current_data) > 5:
                print(f"   ... and {len(current_data) - 5} more districts")
            
            # Save current data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            data_file = os.path.join(self.data_dir, f"rainfall_data_{timestamp}.json")
            fetcher.save_data(current_data, f"rainfall_data_{timestamp}.json")
            
            # Generate historical simulation
            print("Generating historical rainfall simulation...")
            historical_df = fetcher.get_historical_simulation(hours=6)
            
            historical_file = os.path.join(self.data_dir, f"historical_rainfall_{timestamp}.csv")
            historical_df.to_csv(historical_file, index=False, encoding='utf-8')
            print(f"✓ Historical data saved: {len(historical_df)} records")
            
            return current_data, historical_df
            
        except Exception as e:
            print(f"❌ Data fetching failed: {e}")
            return None, None
    
    def demonstrate_artistic_visualization(self, demo_mode=True):
        """
        Demonstrate artistic visualization capabilities.
        
        Args:
            demo_mode: If True, use simulated data for demonstration
        """
        print("\n🎨 Creating Artistic Visualizations")
        print("-" * 40)
        
        try:
            # Initialize visualizer
            visualizer = ArtisticRainfallVisualizer(style='artistic')
            
            # Create static visualization
            print("Creating static artistic visualization...")
            static_path = os.path.join(self.viz_dir, "hk_rainfall_artistic.png")
            fig = visualizer.create_static_visualization(save_path=static_path)
            print(f"✓ Static visualization saved: {static_path}")
            
            # Create animated visualization (shorter for demo)
            print("Creating animated visualization...")
            animation_path = os.path.join(self.viz_dir, "hk_rainfall_animation.gif")
            
            try:
                anim = visualizer.create_animated_visualization(
                    duration=15,  # 15 seconds for demo
                    save_path=animation_path
                )
                print(f"✓ Animation saved: {animation_path}")
            except Exception as e:
                print(f"⚠ Animation creation skipped: {e}")
            
            return True
            
        except Exception as e:
            print(f"❌ Visualization failed: {e}")
            return False
    
    def demonstrate_artistic_effects(self):
        """
        Demonstrate advanced artistic effects.
        """
        print("\n✨ Demonstrating Artistic Effects")
        print("-" * 40)
        
        try:
            from artistic_effects import demonstrate_artistic_effects
            demonstrate_artistic_effects()
            return True
        except Exception as e:
            print(f"❌ Artistic effects demonstration failed: {e}")
            return False
    
    def run_real_time_demo(self, duration_minutes=5):
        """
        Run real-time visualization demo.
        
        Args:
            duration_minutes: How long to run the demo
        """
        print(f"\n🔴 Starting Real-Time Demo ({duration_minutes} minutes)")
        print("-" * 40)
        
        try:
            # Initialize real-time stream
            stream = RealTimeDataStream(update_interval=30)  # 30 seconds
            
            print("Real-time rainfall data stream starting...")
            print("Press Ctrl+C to stop")
            
            start_time = time.time()
            end_time = start_time + (duration_minutes * 60)
            
            while time.time() < end_time:
                try:
                    # Fetch and display current data
                    current_data = stream.fetcher.fetch_current_rainfall_data()
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    
                    print(f"\n[{timestamp}] Current Rainfall Status:")
                    
                    # Show top 3 districts with highest rainfall
                    sorted_districts = sorted(current_data.items(), 
                                            key=lambda x: x[1]['rainfall_mm'], 
                                            reverse=True)[:3]
                    
                    for district, info in sorted_districts:
                        print(f"  {district}: {info['rainfall_mm']} mm")
                    
                    # Brief pause
                    time.sleep(30)
                    
                except KeyboardInterrupt:
                    print("\n⏹ Real-time demo stopped by user")
                    break
                except Exception as e:
                    print(f"Error in real-time demo: {e}")
                    break
            
            print("✓ Real-time demo completed")
            return True
            
        except Exception as e:
            print(f"❌ Real-time demo failed: {e}")
            return False
    
    def run_full_demo(self):
        """
        Run complete demonstration of all features.
        """
        print("🚀 Hong Kong Rainfall Artistic Visualization - Full Demo")
        print("=" * 60)
        
        success_count = 0
        total_demos = 4
        
        # Check dependencies first
        if not self.check_dependencies():
            print("\n⚠ Continuing with limited functionality...")
        
        # Demo 1: Data fetching
        try:
            current_data, historical_df = self.demonstrate_data_fetching()
            if current_data is not None:
                success_count += 1
        except Exception as e:
            print(f"Data fetching demo failed: {e}")
        
        # Demo 2: Artistic visualization
        try:
            if self.demonstrate_artistic_visualization():
                success_count += 1
        except Exception as e:
            print(f"Visualization demo failed: {e}")
        
        # Demo 3: Artistic effects
        try:
            if self.demonstrate_artistic_effects():
                success_count += 1
        except Exception as e:
            print(f"Artistic effects demo failed: {e}")
        
        # Demo 4: Real-time (short demo)
        try:
            print(f"\n🔴 Real-time demo starting (2 minutes)...")
            if self.run_real_time_demo(duration_minutes=2):
                success_count += 1
        except Exception as e:
            print(f"Real-time demo failed: {e}")
        
        # Summary
        print(f"\n📊 Demo Summary")
        print(f"   Successful demos: {success_count}/{total_demos}")
        print(f"   Generated files in: {self.viz_dir}")
        print(f"   Data files in: {self.data_dir}")
        
        if success_count >= total_demos // 2:
            print("✅ Demo completed successfully!")
        else:
            print("⚠ Demo completed with some limitations")
            print("   Please check dependencies and try again")
        
        return success_count


def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Hong Kong Rainfall Artistic Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --demo                    # Run full demonstration
  python main.py --visualize              # Create static visualization
  python main.py --animate --duration 30  # Create 30-second animation
  python main.py --realtime --minutes 10  # Run real-time for 10 minutes
        """
    )
    
    parser.add_argument('--demo', action='store_true',
                       help='Run full demonstration of all features')
    parser.add_argument('--visualize', action='store_true',
                       help='Create static artistic visualization')
    parser.add_argument('--animate', action='store_true',
                       help='Create animated visualization')
    parser.add_argument('--duration', type=int, default=30,
                       help='Animation duration in seconds (default: 30)')
    parser.add_argument('--realtime', action='store_true',
                       help='Run real-time visualization')
    parser.add_argument('--minutes', type=int, default=5,
                       help='Real-time duration in minutes (default: 5)')
    parser.add_argument('--style', choices=['artistic', 'natural', 'neon'], 
                       default='artistic',
                       help='Visualization style (default: artistic)')
    
    return parser.parse_args()


def main():
    """
    Main entry point for the Hong Kong Rainfall Art Project.
    """
    args = parse_arguments()
    
    # Initialize project
    project = HKRainfallArtProject()
    
    # Run based on arguments
    if args.demo:
        project.run_full_demo()
    
    elif args.visualize:
        print("🎨 Creating static visualization...")
        project.demonstrate_artistic_visualization()
    
    elif args.animate:
        print(f"🎬 Creating {args.duration}-second animation...")
        try:
            visualizer = ArtisticRainfallVisualizer(style=args.style)
            animation_path = os.path.join(project.viz_dir, 
                                        f"rainfall_animation_{args.style}.gif")
            visualizer.create_animated_visualization(
                duration=args.duration, 
                save_path=animation_path
            )
            print(f"✓ Animation saved: {animation_path}")
        except Exception as e:
            print(f"❌ Animation failed: {e}")
    
    elif args.realtime:
        print(f"🔴 Starting real-time visualization ({args.minutes} minutes)...")
        project.run_real_time_demo(duration_minutes=args.minutes)
    
    else:
        # Default: run demo
        print("No specific mode selected. Running full demo...")
        project.run_full_demo()


if __name__ == "__main__":
    main()
