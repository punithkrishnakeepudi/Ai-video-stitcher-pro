#!/usr/bin/env python3
"""
AI Video Stitcher Pro - Setup Script
Automatically sets up the application environment
"""

import os
import sys
import subprocess
import platform


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"📋 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {sys.version}")
        return False
    print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def install_system_dependencies():
    """Install system-level dependencies"""
    system = platform.system().lower()

    if system == "linux":
        print("🐧 Detected Linux system")
        commands = [
            "sudo apt-get update",
            "sudo apt-get install -y python3-pip python3-venv",
            "sudo apt-get install -y ffmpeg",
            "sudo apt-get install -y libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1"
        ]

        for cmd in commands:
            if not run_command(cmd, f"Installing system dependencies: {cmd}"):
                print("⚠️  Some system dependencies might not be installed. Continuing...")

    elif system == "darwin":  # macOS
        print("🍎 Detected macOS system")
        if not run_command("which brew", "Checking for Homebrew"):
            print("❌ Homebrew not found. Please install Homebrew first:")
            print(
                "   /bin/bash -c \"$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\"")
            return False

        commands = [
            "brew install python3",
            "brew install ffmpeg"
        ]

        for cmd in commands:
            run_command(cmd, f"Installing dependencies: {cmd}")

    elif system == "windows":
        print("🪟 Detected Windows system")
        print("⚠️  Please ensure you have:")
        print("   - Python 3.8+ installed")
        print("   - FFmpeg installed and in PATH")
        print("   - Visual Studio Build Tools (for some packages)")

    return True


def create_project_structure():
    """Create necessary directories"""
    directories = [
        "uploads",
        "outputs",
        "temp",
        "static",
        "templates"
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"📁 Created directory: {directory}")

    return True


def create_virtual_environment():
    """Create and activate virtual environment"""
    if not run_command("python3 -m venv venv", "Creating virtual environment"):
        return False

    # Determine activation command based on OS
    if platform.system().lower() == "windows":
        activate_cmd = "venv\\Scripts\\activate"
        pip_cmd = "venv\\Scripts\\pip"
    else:
        activate_cmd = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"

    print(f"✅ Virtual environment created")
    print(f"💡 To activate: {activate_cmd}")

    return pip_cmd


def install_python_dependencies(pip_cmd):
    """Install Python packages"""
    # Install core requirements
    requirements = [
        "flask==2.3.3",
        "flask-cors==4.0.0",
        "opencv-python==4.8.1.78",
        "numpy==1.24.3",
        "werkzeug==2.3.7",
        "pillow==10.0.0",
        "requests==2.31.0"
    ]

    print("📦 Installing core requirements...")
    for req in requirements:
        if not run_command(f"{pip_cmd} install {req}", f"Installing {req}"):
            print(f"⚠️  Failed to install {req}, continuing...")

    # Install optional ML dependencies
    ml_requirements = [
        "scikit-learn==1.3.0",
        "tensorflow==2.13.0"
    ]

    print("🤖 Installing ML dependencies (optional)...")
    for req in ml_requirements:
        if not run_command(f"{pip_cmd} install {req}", f"Installing {req}"):
            print(f"⚠️  Failed to install {req}. ML features may be limited.")

    # Install FFmpeg Python wrapper
    run_command(f"{pip_cmd} install ffmpeg-python", "Installing ffmpeg-python")

    return True


def create_app_files():
    """Create main application files"""

    # Create app.py (main Flask application)
    app_content = '''# This file should contain the Flask backend code
# Copy the Flask backend code from the artifacts here
print("Please copy the Flask backend code to this file")
'''

    with open("app.py", "w") as f:
        f.write(app_content)

    # Create index.html (frontend)
    html_content = '''<!-- This file should contain the HTML frontend -->
<!-- Copy the HTML web application code from the artifacts here -->
<!DOCTYPE html>
<html>
<head><title>AI Video Stitcher Pro</title></head>
<body><h1>Please copy the HTML code from artifacts</h1></body>
</html>
'''

    with open("index.html", "w") as f:
        f.write(html_content)

    print("📄 Created placeholder app files")
    print("⚠️  Please copy the actual code from the artifacts:")
    print("   - Copy Flask backend code to app.py")
    print("   - Copy HTML frontend code to index.html")

    return True


def create_run_script():
    """Create run script for easy startup"""
    if platform.system().lower() == "windows":
        script_content = '''@echo off
echo Starting AI Video Stitcher Pro...
call venv\\Scripts\\activate
python app.py
pause
'''
        with open("run.bat", "w") as f:
            f.write(script_content)
        print("📄 Created run.bat script")

    else:
        script_content = '''#!/bin/bash
echo "Starting AI Video Stitcher Pro..."
source venv/bin/activate
python3 app.py
'''
        with open("run.sh", "w") as f:
            f.write(script_content)
        os.chmod("run.sh", 0o755)
        print("📄 Created run.sh script")


def main():
    """Main setup function"""
    print("🎬 AI Video Stitcher Pro - Setup Script")
    print("=" * 50)

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Install system dependencies
    print("\n📋 Installing system dependencies...")
    install_system_dependencies()

    # Create project structure
    print("\n📁 Creating project structure...")
    create_project_structure()

    # Create virtual environment
    print("\n🏠 Setting up virtual environment...")
    pip_cmd = create_virtual_environment()
    if not pip_cmd:
        print("❌ Failed to create virtual environment")
        sys.exit(1)

    # Install Python dependencies
    print("\n📦 Installing Python dependencies...")
    install_python_dependencies(pip_cmd)

    # Create application files
    print("\n📄 Creating application files...")
    create_app_files()

    # Create run script
    print("\n🚀 Creating run script...")
    create_run_script()

    print("\n" + "=" * 50)
    print("✅ Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Copy the Flask backend code to app.py")
    print("2. Copy the HTML frontend code to index.html")
    print("3. Run the application:")
    if platform.system().lower() == "windows":
        print("   - Double-click run.bat")
        print("   - Or: run.bat")
    else:
        print("   - ./run.sh")
        print("   - Or: source venv/bin/activate && python3 app.py")
    print("\n🌐 Access the web app at: http://localhost:5000")
    print("\n💡 For help, check the documentation or create an issue on GitHub")


if __name__ == "__main__":
    main()