"""
Automated Setup Script for Named Entity Linking System
Installs dependencies and sets up the environment
"""

import subprocess
import sys
import os
import platform
from pathlib import Path

def run_command(command, description=""):
    """Run a command and handle errors"""
    print(f"{'='*50}")
    if description:
        print(f"🔧 {description}")
    print(f"📝 Running: {command}")
    print(f"{'='*50}")
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print("✅ Success!")
        if result.stdout:
            print("Output:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print("Output:", e.stdout)
        if e.stderr:
            print("Error output:", e.stderr)
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"❌ Python {version.major}.{version.minor} detected")
        print("⚠️ This system requires Python 3.8 or higher")
        print("Please upgrade Python and try again")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Compatible!")
    return True

def install_requirements():
    """Install Python requirements"""
    print("\n📦 Installing Python packages...")
    
    # Check if requirements.txt exists
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found!")
        print("Please ensure requirements.txt is in the current directory")
        return False
    
    # Upgrade pip first
    success = run_command(
        f"{sys.executable} -m pip install --upgrade pip",
        "Upgrading pip"
    )
    if not success:
        print("⚠️ Warning: Could not upgrade pip, continuing anyway...")
    
    # Install requirements
    success = run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing required packages"
    )
    return success

def download_spacy_model():
    """Download required spaCy model"""
    print("\n🧠 Setting up spaCy language model...")
    
    success = run_command(
        f"{sys.executable} -m spacy download en_core_web_sm",
        "Downloading English language model"
    )
    
    if not success:
        print("⚠️ Failed to download spaCy model")
        print("You can try manually later with:")
        print("python -m spacy download en_core_web_sm")
        return False
    
    return True

def test_installation():
    """Test if installation was successful"""
    print("\n🧪 Testing installation...")
    
    test_code = """
import spacy
import pandas as pd
import numpy as np
import requests
import json

# Test spaCy model loading
try:
    nlp = spacy.load('en_core_web_sm')
    doc = nlp('Apple Inc. is a technology company based in Cupertino.')
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    print('✅ spaCy working - Found entities:', entities)
except Exception as e:
    print('❌ spaCy test failed:', str(e))
    exit(1)

# Test other critical imports
try:
    import sklearn
    print('✅ scikit-learn imported successfully')
except ImportError as e:
    print('❌ scikit-learn import failed:', str(e))
    exit(1)

try:
    import transformers
    print('✅ transformers imported successfully')
except ImportError as e:
    print('❌ transformers import failed:', str(e))
    exit(1)

print('🎉 All tests passed! Installation successful.')
"""
    
    # Write test code to temporary file
    test_file = Path("test_installation.py")
    try:
        with open(test_file, 'w') as f:
            f.write(test_code)
        
        # Run the test
        success = run_command(
            f"{sys.executable} test_installation.py",
            "Running installation tests"
        )
        
        # Clean up test file
        test_file.unlink()
        
        return success
        
    except Exception as e:
        print(f"❌ Failed to create/run test file: {e}")
        return False

def create_directory_structure():
    """Create necessary directories for the project"""
    print("\n📁 Creating directory structure...")
    
    directories = [
        "data",
        "models",
        "outputs",
        "logs",
        "cache",
        "configs"
    ]
    
    for directory in directories:
        try:
            Path(directory).mkdir(exist_ok=True)
            print(f"✅ Created/verified directory: {directory}")
        except Exception as e:
            print(f"❌ Failed to create directory {directory}: {e}")
            return False
    
    return True

def setup_environment_file():
    """Create a sample environment configuration file"""
    print("\n⚙️ Setting up environment configuration...")
    
    env_content = """# Named Entity Linking System Configuration
# Copy this file to .env and modify as needed

# API Keys (add your keys here)
WIKIDATA_API_TIMEOUT=30
MAX_CONCURRENT_REQUESTS=10

# Model Configuration
SPACY_MODEL=en_core_web_sm
SIMILARITY_THRESHOLD=0.8
MAX_CANDIDATES=50

# Cache Settings
ENABLE_CACHE=true
CACHE_EXPIRY_DAYS=7

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/nel_system.log

# Performance
BATCH_SIZE=32
MAX_WORKERS=4
"""
    
    env_file = Path("env.example")
    try:
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("✅ Created env.example file")
        print("📝 Copy this to .env and configure as needed")
        return True
    except Exception as e:
        print(f"❌ Failed to create env.example: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Named Entity Linking System Setup")
    print("====================================")
    
    # Check Python version first
    if not check_python_version():
        sys.exit(1)
    
    # Create directory structure
    if not create_directory_structure():
        print("❌ Failed to create directory structure")
        sys.exit(1)
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements")
        sys.exit(1)
    
    # Download spaCy model
    if not download_spacy_model():
        print("❌ Failed to download spaCy model")
        print("⚠️ You may need to install it manually later")
    
    # Test installation
    if not test_installation():
        print("❌ Installation tests failed")
        print("⚠️ Please check the errors above and try again")
        sys.exit(1)
    
    # Setup environment file
    setup_environment_file()
    
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE! 🎉")
    print("="*60)
    print("✅ All dependencies installed successfully")
    print("✅ Directory structure created")
    print("✅ Environment template created")
    print("\n📋 Next steps:")
    print("1. Copy env.example to .env and configure your settings")
    print("2. Add your API keys to the .env file")
    print("3. Run your Named Entity Linking system!")
    print("\n🚀 Happy linking!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error during setup: {e}")
        sys.exit(1)