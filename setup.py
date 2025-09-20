#!/usr/bin/env python3
"""
Setup and Installation Script for Hong Kong Rainfall Art Project
===============================================================

This script helps set up the project environment and install dependencies.
"""

import subprocess
import sys
import os
from pathlib import Path


def check_python_version():
    """Check if Python version is sufficient."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    else:
        print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
        return True


def install_requirements():
    """Install required packages."""
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("✓ All packages installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Package installation failed: {e}")
        return False


def create_directories():
    """Create necessary project directories."""
    project_root = Path(__file__).parent
    directories = ['data', 'visualizations', 'assets']
    
    print("📁 Creating project directories...")
    for dir_name in directories:
        dir_path = project_root / dir_name
        dir_path.mkdir(exist_ok=True)
        print(f"   ✓ {dir_name}/")
    
    return True


def test_imports():
    """Test if all required modules can be imported."""
    print("🔍 Testing module imports...")
    
    required_modules = [
        'matplotlib', 'numpy', 'pandas', 'requests', 
        'bs4', 'scipy', 'PIL'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"   ✓ {module}")
        except ImportError as e:
            print(f"   ❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n⚠ Failed to import: {', '.join(failed_imports)}")
        print("   Try running: pip install -r requirements.txt")
        return False
    else:
        print("✓ All modules imported successfully")
        return True


def run_basic_test():
    """Run a basic functionality test."""
    print("🧪 Running basic functionality test...")
    
    try:
        # Test basic imports from project modules
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        
        # Test if we can create sample data
        import numpy as np
        x = np.linspace(0, 10, 50)
        y = np.linspace(0, 8, 40)
        X, Y = np.meshgrid(x, y)
        rainfall = 20 * np.exp(-((X-5)**2 + (Y-4)**2) / 4)
        
        print(f"   ✓ Sample data created: {rainfall.shape}")
        
        # Test matplotlib
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.contourf(X, Y, rainfall, levels=10, cmap='Blues')
        ax.set_title('Test Rainfall Pattern')
        
        # Save test image
        test_image_path = Path(__file__).parent / "visualizations" / "setup_test.png"
        fig.savefig(test_image_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"   ✓ Test visualization saved: {test_image_path}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Basic test failed: {e}")
        return False


def main():
    """Main setup function."""
    print("🚀 Hong Kong Rainfall Art Project Setup")
    print("=" * 50)
    
    success_count = 0
    total_steps = 5
    
    # Step 1: Check Python version
    if check_python_version():
        success_count += 1
    
    # Step 2: Create directories
    if create_directories():
        success_count += 1
    
    # Step 3: Install requirements
    if install_requirements():
        success_count += 1
    
    # Step 4: Test imports
    if test_imports():
        success_count += 1
    
    # Step 5: Run basic test
    if run_basic_test():
        success_count += 1
    
    print("\n" + "=" * 50)
    print(f"Setup Summary: {success_count}/{total_steps} steps completed")
    
    if success_count == total_steps:
        print("✅ Setup completed successfully!")
        print("\nNext steps:")
        print("  1. Run demo: python main.py --demo")
        print("  2. Create visualization: python main.py --visualize")
        print("  3. Check README.md for more options")
    else:
        print("⚠ Setup completed with issues")
        print("   Please check error messages above and retry")
    
    return success_count == total_steps


if __name__ == "__main__":
    main()
