"""
Bootstrap script: Install Miniforge (conda) or set up Python 3.10 venv + pip.
Run this once to set up the trading environment.
"""
import subprocess
import sys
import os
import shutil
from pathlib import Path

def check_conda_installed():
    """Check if conda is already installed."""
    try:
        subprocess.run(['conda', '--version'], capture_output=True, check=True)
        print("✓ conda is already installed")
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False

def install_miniforge():
    """Download and install Miniforge (conda) on Windows."""
    print("\n" + "="*60)
    print("Installing Miniforge (conda)...")
    print("="*60)
    
    # Download Miniforge installer
    miniforge_url = "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe"
    installer_path = Path(os.environ['TEMP']) / 'Miniforge3-Windows-x86_64.exe'
    
    print(f"\nDownloading Miniforge from {miniforge_url}...")
    try:
        import urllib.request
        urllib.request.urlretrieve(miniforge_url, str(installer_path))
        print("✓ Download complete")
    except Exception as e:
        print(f"✗ Download failed: {e}")
        print("\nManual installation:")
        print(f"  1. Download: {miniforge_url}")
        print("  2. Run the installer")
        print("  3. Close and reopen PowerShell")
        print("  4. Run: python bootstrap.py")
        return False
    
    # Run installer
    print("\nRunning Miniforge installer...")
    print("(This will open an installer window. Follow the prompts and accept defaults.)")
    try:
        subprocess.run([
            str(installer_path),
            '/InstallationType=JustMe',
            '/AddToPath=1',
            '/RegisterPython=0',
            '/S'
        ], check=True)
        print("✓ Miniforge installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Installation failed: {e}")
        return False

def setup_conda_env():
    """Create conda env from environment.yml."""
    print("\n" + "="*60)
    print("Creating conda environment from environment.yml...")
    print("="*60)
    
    try:
        subprocess.run(['conda', 'env', 'create', '-f', 'environment.yml', '-y'], check=True)
        print("\n✓ Conda environment created successfully!")
        print("\nTo activate the environment and run the GUI:")
        print("  conda activate trading")
        print("  python scripts/gui_app.py")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to create conda env: {e}")
        return False
    except FileNotFoundError:
        print("✗ conda command not found. Please reopen PowerShell and try again.")
        return False

def setup_venv_fallback():
    """Fallback: create venv with Python 3.10 and install from requirements.txt."""
    print("\n" + "="*60)
    print("Setting up Python venv + pip (fallback)...")
    print("="*60)
    
    # Check if Python 3.10 available
    try:
        result = subprocess.run(['py', '-3.10', '--version'], capture_output=True, text=True, check=True)
        print(f"✓ Python found: {result.stdout.strip()}")
        python_exe = 'py -3.10'
    except subprocess.CalledProcessError:
        print("✗ Python 3.10 not found.")
        print("\nPlease install Python 3.10 from https://www.python.org/")
        print("Then run this script again.")
        return False
    
    venv_path = Path('.venv')
    if venv_path.exists():
        print(f"Note: {venv_path} already exists, skipping creation")
    else:
        print(f"\nCreating virtual environment at {venv_path}...")
        try:
            subprocess.run([python_exe, '-m', 'venv', '.venv'], check=True)
            print(f"✓ Virtual environment created")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to create venv: {e}")
            return False
    
    # Install packages
    print("\nInstalling packages from requirements.txt...")
    venv_python = venv_path / 'Scripts' / 'python.exe'
    try:
        subprocess.run([str(venv_python), '-m', 'pip', 'install', '--upgrade', 'pip'], check=True)
        subprocess.run([str(venv_python), '-m', 'pip', 'install', '-r', 'requirements.txt'], check=True)
        print("✓ Packages installed successfully!")
        print("\nTo run the GUI:")
        print(f"  .\\venv\\Scripts\\Activate.ps1")
        print("  python scripts/gui_app.py")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install packages: {e}")
        return False

def main():
    print("="*60)
    print("Trading System ML - Environment Setup Bootstrap")
    print("="*60)
    
    # Check current directory
    if not os.path.exists('environment.yml'):
        print("\n✗ Error: environment.yml not found in current directory")
        print("Please run this script from the repository root")
        sys.exit(1)
    
    # Option 1: Try conda
    if check_conda_installed():
        print("\nUsing existing conda installation...")
        if setup_conda_env():
            return
    else:
        print("\nconda not found. Options:")
        print("  A) Install Miniforge (recommended) [y/n]")
        response = input("  > ").strip().lower()
        
        if response == 'y':
            if install_miniforge():
                print("\n✓ Miniforge installed!")
                print("\nPlease close and reopen PowerShell, then run this script again.")
                sys.exit(0)
    
    # Fallback: venv + pip
    print("\nFalling back to Python venv + pip...")
    if setup_venv_fallback():
        return
    
    print("\n✗ Setup failed. Please try:")
    print("  1. Install Miniforge manually: https://github.com/conda-forge/miniforge/releases")
    print("  2. Or install Python 3.10 and try again")
    sys.exit(1)

if __name__ == '__main__':
    main()
