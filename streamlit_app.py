"""Entry point for Streamlit Community Cloud (main file path: streamlit_app.py)."""

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))
# run_path re-executes the app on every Streamlit rerun, which a plain import would not.
runpy.run_path(str(ROOT / "src" / "ragkit" / "ui" / "app.py"), run_name="__main__")
