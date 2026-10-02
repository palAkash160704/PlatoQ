# Development Setup

## Prerequisites

| Tool | Version | Required? |
|---|---|---|
| Python | 3.11+ | ✅ Yes |
| SUMO | 1.18+ | ✅ Yes (for simulation phases) |
| Git | any | ✅ Yes |
| pip / venv | bundled with Python | ✅ Yes |

---

## 1. Python Installation

Download and install Python 3.11 or later from [python.org](https://www.python.org/downloads/).

Verify:

```bash
python --version   # Should print Python 3.11.x or higher
```

---

## 2. Virtual Environment

Create and activate a project-specific virtual environment:

```bash
# Create
python -m venv .venv

# Activate — Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Activate — Windows (cmd)
.venv\Scripts\activate.bat

# Activate — Linux / macOS
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
# Production dependencies only
pip install -r requirements.txt

# Development dependencies (includes testing and linting)
pip install -r requirements-dev.txt
```

---

## 4. SUMO Installation

SUMO (Simulation of Urban Mobility) is required for the traffic simulation phases.

### Option A: pip (recommended — tested on this project)

This installs SUMO, TraCI, and sumolib directly into the virtual environment:

```bash
pip install eclipse-sumo traci sumolib
```

Verify installation:

```bash
python -c "import sumolib; print(sumolib.checkBinary('sumo'))"
# Should print the path to sumo.exe
```

This approach was tested on Windows 10/11 with Python 3.13 and installs
SUMO 1.27.1. The SUMO binaries are placed inside the Python package
directory and are found automatically via `sumolib.checkBinary()`.

**Note:** With this method, `SUMO_HOME` does NOT need to be set.

### Option B: System installer (Windows)

Download the installer from [sumo.dlr.de/docs/Downloads.php](https://sumo.dlr.de/docs/Downloads.php) and run it.

After installation, set `SUMO_HOME`:

```powershell
# Windows — System Properties → Environment Variables
# Variable: SUMO_HOME
# Value:    <your SUMO install path>   (e.g. C:\Program Files (x86)\Eclipse\Sumo)
```

### Option C: Linux (Ubuntu/Debian)

```bash
sudo add-apt-repository ppa:sumo/stable
sudo apt update
sudo apt install sumo sumo-tools sumo-doc
export SUMO_HOME=/usr/share/sumo
```

### Option D: macOS (Homebrew)

```bash
brew install sumo
export SUMO_HOME=/opt/homebrew/share/sumo
```

### Verify SUMO

```bash
# If installed via pip:
python -c "import sumolib; print(sumolib.checkBinary('sumo'))"

# If installed via system installer:
sumo --version
```

---

## 5. SUMO_HOME Setup (only for system installs)

If you installed SUMO via pip (Option A), `SUMO_HOME` is **not required**.

If you installed SUMO via the system installer, set the `SUMO_HOME`
environment variable:

### Option A: `.env` file (recommended for development)

```bash
cp .env.example .env
# Edit .env and set:
# SUMO_HOME=/usr/share/sumo                        (Linux)
# SUMO_HOME=/opt/homebrew/share/sumo               (macOS)
```

### Option B: System environment variable

```bash
# Linux / macOS — add to ~/.bashrc or ~/.zshrc
export SUMO_HOME=/usr/share/sumo

# Windows — System Properties → Environment Variables
```

### Verify

```bash
echo $SUMO_HOME          # Linux/macOS
echo %SUMO_HOME%         # Windows cmd
echo $env:SUMO_HOME      # Windows PowerShell
```

---

## 6. Running Tests

```bash
# Run all tests (unit + integration)
pytest

# Run with verbose output
pytest -v

# Run only unit tests (no SUMO required)
pytest -v -k "not Integration"

# Run only integration tests (requires SUMO)
pytest -v -k "Integration"

# Run a specific test file
pytest tests/test_vehicle.py

# Run a specific test class or method
pytest tests/test_vehicle.py::TestVehicleStateCreation::test_minimal_creation
```

**Note:** Integration tests are automatically skipped with a clear
message if SUMO is not installed.

---

## 7. Running the Application

```bash
# Headless SUMO simulation (default)
python -m platooning.main

# With SUMO-GUI for visual inspection
python -m platooning.main --gui
```

Expected output (Phase 1):

```
==================================================
Hybrid Quantum-Classical Vehicle Platooning
v0.1.0
==================================================

<timestamp> | INFO | Configuration loaded successfully.
<timestamp> | INFO | Configuration validated — no errors.
<timestamp> | INFO | Simulation backend: SUMO
<timestamp> | INFO | Starting SUMO simulation...
<timestamp> | INFO | Starting SUMO with command: ...
<timestamp> | INFO | TraCI connection established.
<timestamp> | INFO | SUMO simulation started (gui=False).
<timestamp> | INFO | [SIM] t=5.0s | vehicles=3
<timestamp> | INFO |   V1   | x=  ... | y= ... | speed=... | accel=... | lane=main_road_0
...
<timestamp> | INFO | Simulation completed. Total steps: 809 | Final time: 80.9s
<timestamp> | INFO | TraCI connection closed.
<timestamp> | INFO | SUMO simulation stopped after 809 steps.
```

---

## 8. Code Quality

```bash
# Format code with black
black src/ tests/

# Lint with ruff
ruff check src/ tests/

# Auto-fix ruff violations
ruff check --fix src/ tests/

# Type checking (optional, strict mode disabled)
mypy src/
```

---

## 9. Project Layout

See [architecture.md](architecture.md) for a detailed breakdown of every module.
