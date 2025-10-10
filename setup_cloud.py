#!/usr/bin/env python3
"""
Setup script for Streamlit Cloud deployment.
This script handles system dependencies and environment setup.
"""

import os
import sys
import subprocess

def install_system_dependencies():
    """Install system dependencies for RDKit molecular drawing."""
    try:
        # Try to install system packages (this may not work on all systems)
        packages = [
            "libxrender1",
            "libxext6", 
            "libx11-6",
            "libfreetype6",
            "libfontconfig1"
        ]
        
        for package in packages:
            try:
                subprocess.run(["apt-get", "update"], check=True, capture_output=True)
                subprocess.run(["apt-get", "install", "-y", package], check=True, capture_output=True)
                print(f"✅ Installed {package}")
            except subprocess.CalledProcessError:
                print(f"⚠️  Could not install {package} (may not be available)")
                
    except Exception as e:
        print(f"⚠️  System dependency installation failed: {e}")
        print("Continuing without system dependencies...")

def setup_environment():
    """Set up environment variables for cloud deployment."""
    # Set environment variables to handle missing libraries gracefully
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    os.environ["MPLBACKEND"] = "Agg"
    
    # Disable matplotlib GUI
    import matplotlib
    matplotlib.use('Agg')
    
    print("✅ Environment configured for cloud deployment")

def test_imports():
    """Test critical imports and provide fallbacks."""
    print("🧪 Testing imports...")
    
    # Test RDKit
    try:
        from rdkit import Chem
        print("✅ RDKit basic import successful")
        
        # Test drawing (may fail on cloud)
        try:
            from rdkit.Chem import Draw
            print("✅ RDKit drawing import successful")
        except ImportError as e:
            print(f"⚠️  RDKit drawing not available: {e}")
            print("Molecular highlights will be disabled")
            
    except ImportError as e:
        print(f"❌ RDKit import failed: {e}")
        sys.exit(1)
    
    # Test other critical imports
    try:
        import torch
        import transformers
        import streamlit
        import plotly
        print("✅ All critical imports successful")
    except ImportError as e:
        print(f"❌ Critical import failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("🚀 Setting up Synapse.AI for cloud deployment...")
    
    # Only try to install system dependencies if we have permission
    if os.geteuid() == 0:  # Running as root
        install_system_dependencies()
    else:
        print("ℹ️  Skipping system dependency installation (not running as root)")
    
    setup_environment()
    test_imports()
    
    print("✅ Setup complete! Ready for cloud deployment.")
