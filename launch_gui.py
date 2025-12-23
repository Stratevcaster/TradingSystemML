"""
Auto-setup launcher for the Trading GUI.
Checks for required dependencies and auto-installs them if missing.
Then launches the GUI.
"""
import subprocess
import sys
import os

REQUIRED_PACKAGES = {
    'tensorflow': 'tensorflow',
    'pandas': 'pandas',
    'matplotlib': 'matplotlib',
    'PIL': 'pillow',
    'numpy': 'numpy',
    'sklearn': 'scikit-learn',
}

MIN_PYTHON_VERSION = (3, 10)
MAX_PYTHON_VERSION = (3, 11)  # TensorFlow typically supports up to 3.11

def check_python_version():
    """Warn user if Python version is incompatible with TensorFlow."""
    version_info = sys.version_info
    if version_info[:2] > MAX_PYTHON_VERSION:
        print(f"WARNING: Python {version_info.major}.{version_info.minor} detected.")
        print(f"TensorFlow may not have wheels for Python > {MAX_PYTHON_VERSION[0]}.{MAX_PYTHON_VERSION[1]}.")
        print("\nRecommendation: Use conda to create a Python 3.10/3.11 environment:")
        print("  conda create -n trading -y --file environment.yml")
        print("  conda activate trading")
        print("  python scripts/gui_app.py")
        return False
    return True

def check_and_install_packages():
    """Check if required packages are installed; if not, install them."""
    missing = []
    
    for import_name, package_name in REQUIRED_PACKAGES.items():
        try:
            __import__(import_name)
        except ImportError:
            missing.append(package_name)
    
    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("\nAttempting to install using pip...")
        
        # Try to install all missing packages at once
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing)
            print("\nPackages installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"\nFailed to install packages: {e}")
            print("\nAlternative: Use conda to create the environment:")
            print("  conda create -n trading -y --file environment.yml")
            print("  conda activate trading")
            print("  python scripts/gui_app.py")
            return False
    else:
        print("All required packages are installed.")
        return True

def main():
    print("=" * 60)
    print("Trading GUI - Auto Setup Launcher")
    print("=" * 60)
    print()
    
    # Check Python version first
    if not check_python_version():
        sys.exit(1)
    
    # Try to install missing packages
    if not check_and_install_packages():
        sys.exit(1)
    
    print("\nLaunching GUI...")
    print("=" * 60)
    
    # Import and run the GUI
    try:
        os.chdir(os.path.dirname(os.path.abspath(__file__)) or '.')
        from scripts import gui_app
        app = gui_app.App()
        app.mainloop()
    except Exception as e:
        print(f"Error launching GUI: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
