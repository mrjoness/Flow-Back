import os

# Get the absolute path to the parent directory of this file (the repo root)
FLOWBACK_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

FLOWBACK_SRC = f"{FLOWBACK_BASE}/src"
FLOWBACK_SCRIPTS = f"{FLOWBACK_SRC}/scripts"
FLOWBACK_JOBDIR = f"{FLOWBACK_BASE}/jobs"
FLOWBACK_MODELS = f"{FLOWBACK_BASE}/models"
FLOWBACK_OUTPUTS = f"{FLOWBACK_BASE}/outputs"
FLOWBACK_DATA = f"{FLOWBACK_BASE}/data"
FLOWBACK_FF = f"{FLOWBACK_BASE}/forcefield"
FLOWBACK_INPUTS = f"{FLOWBACK_BASE}/inputs"
