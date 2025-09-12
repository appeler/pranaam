#!/usr/bin/env python3
"""
Simple script to run the Streamlit app locally.
Usage: python run_app.py
"""

import subprocess
import sys
import os

def main():
    """Run the Streamlit app."""
    # Get the directory containing this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(script_dir, "streamlit_app.py")
    
    print("🚀 Starting Pranaam Streamlit App...")
    print(f"📁 App location: {app_path}")
    print("🌐 App will be available at: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop the app")
    print("-" * 50)
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", app_path,
            "--server.headless", "false",
            "--server.enableCORS", "false",
            "--server.enableXsrfProtection", "false"
        ], check=True)
    except KeyboardInterrupt:
        print("\n🛑 App stopped by user")
    except FileNotFoundError:
        print("❌ Error: streamlit not found. Install with: pip install streamlit")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running streamlit: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()