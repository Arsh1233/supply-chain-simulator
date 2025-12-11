import subprocess
import sys
import os

def install_packages():
    """Install packages one by one to avoid dependency conflicts"""
    
    packages = [
        "streamlit==1.28.1",
        "numpy==1.24.4",
        "pandas==2.1.4",
        "scikit-learn==1.3.2",
        "networkx==3.2.1",
        "matplotlib==3.8.2",
        "seaborn==0.13.0",
        "plotly==5.18.0",
        "scipy==1.11.4",
        "gym==0.26.2",
        "stochastic==0.6.0",
        "python-igraph==0.10.8"
    ]
    
    print("Installing packages...")
    
    for package in packages:
        print(f"\nInstalling {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}: {e}")
    
    # Install PyTorch separately (CPU version for compatibility)
    print("\nInstalling PyTorch...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch==2.1.2", "--index-url", "https://download.pytorch.org/whl/cpu"])
        print("✓ PyTorch installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install PyTorch: {e}")
        print("Trying alternative installation...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "torch==2.1.2"])
        except:
            print("Please install PyTorch manually from: https://pytorch.org/get-started/locally/")

if __name__ == "__main__":
    install_packages()