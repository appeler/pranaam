"""Run the Pranaam Streamlit app locally."""

import subprocess
import sys
from pathlib import Path


def main() -> None:
    """Run the Streamlit app."""
    app_path = Path(__file__).resolve().parent / "streamlit_app.py"

    print("🚀 Starting Pranaam Streamlit App...")
    print(f"📁 App location: {app_path}")
    print("🌐 App will be available at: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop the app")
    print("-" * 50)

    try:
        # Run streamlit
        subprocess.run(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(app_path),
                "--server.headless",
                "false",
                "--server.enableCORS",
                "false",
                "--server.enableXsrfProtection",
                "false",
            ],
            check=True,
        )
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
