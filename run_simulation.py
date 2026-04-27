"""
run_simulation.py — Single entry point for the entire project.

Usage:
    python run_simulation.py            # full simulation with dashboard
    python run_simulation.py --novv     # no-V2V comparison run only
    python run_simulation.py --build    # rebuild SUMO network then run
    python run_simulation.py --train    # train ML models only (no SUMO)

What happens:
  1. Builds SUMO network if needed
  2. Trains Random Forest + LSTM
  3. Opens SUMO-GUI + live dashboard
  4. Runs all 4 scenario phases automatically
  5. Generates all reports, graphs, CSV logs
"""

import sys, os, subprocess, argparse

# ── Paths ────────────────────────────────────────────────────
ROOT     = os.path.dirname(os.path.abspath(__file__))
SRC_DIR  = os.path.join(ROOT, 'src')
MAP_DIR  = os.path.join(ROOT, 'map')
NET_FILE = os.path.join(MAP_DIR, 'intersection.net.xml')
NODE_FILE = os.path.join(MAP_DIR, "intersection.nod.xml")
EDGE_FILE = os.path.join(MAP_DIR, "intersection.edg.xml")
CON_FILE  = os.path.join(MAP_DIR, "intersection.con.xml")
SUMO_HOME= r"C:\Program Files (x86)\Eclipse\Sumo"
NETCONV  = os.path.join(SUMO_HOME, 'bin', 'netconvert.exe')

sys.path.insert(0, SRC_DIR)
sys.path.insert(0, os.path.join(SUMO_HOME, 'tools'))

def build_network():
    """Call netconvert to build intersection.net.xml from source files."""
    print("[Setup] Building SUMO network...")
    cmd = [
        NETCONV,
        "--node-files",       os.path.join(MAP_DIR, "intersection.nod.xml"),
        "--edge-files",       os.path.join(MAP_DIR, "intersection.edg.xml"),
        "--connection-files", os.path.join(MAP_DIR, "intersection.con.xml"),
        "--output-file",      NET_FILE,
        "--no-turnarounds",   "true",
        "--tls.default-type", "static",
        "--junctions.corner-detail", "5",
        "--sidewalks.guess",  "true",
        "--crossings.guess",  "true",
        "--geometry.remove",  "true",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("[ERROR] netconvert failed:")
        print(result.stderr)
        sys.exit(1)
    print("[Setup] Network built → intersection.net.xml ✓")


def needs_rebuild():
    if not os.path.exists(NET_FILE):
        return True
    try:
        net_mtime = os.path.getmtime(NET_FILE)
        src_latest = max(os.path.getmtime(NODE_FILE),
                         os.path.getmtime(EDGE_FILE),
                         os.path.getmtime(CON_FILE))
        return src_latest > net_mtime
    except OSError:
        return True

def check_dependencies():
    missing = []
    for pkg in ['numpy','sklearn','matplotlib']:
        try: __import__(pkg)
        except ImportError: missing.append(pkg)
    if missing:
        print(f"[Setup] Installing missing packages: {missing}")
        subprocess.run([sys.executable,'-m','pip','install',
                        'numpy','scikit-learn','matplotlib'], check=True)

def main():
    parser = argparse.ArgumentParser(
        description='Emergency-Focused V2V Intelligence Simulation')
    parser.add_argument('--build',  action='store_true', help='Rebuild SUMO network')
    parser.add_argument('--novv',   action='store_true', help='Run no-V2V comparison only')
    parser.add_argument('--train',  action='store_true', help='Train ML models only')
    args = parser.parse_args()

    print("="*60)
    print("  EMERGENCY-FOCUSED V2V INTELLIGENCE SYSTEM")
    print("  Team: Ch Rakesh | G Adarsh | N JayVardhan")
    print("  Faculty Guide: Mr Naqueeb Ahmed")
    print("="*60)

    # Dependencies
    check_dependencies()

    # Build network if missing or requested
    if args.build or needs_rebuild():
        build_network()
    else:
        print(f"[Setup] Network file found ✓")

    # Train only mode
    if args.train:
        from ml_models import RiskClassifier, TrajectoryPredictor
        rf = RiskClassifier()
        m  = rf.train(8000)
        rf.save(os.path.join(ROOT,'models','rf_model.pkl'))
        lstm = TrajectoryPredictor()
        lstm.train(epochs=8)
        print(f"\nTraining complete. AUC={m['auc']:.4f}  R²={lstm.r2:.4f}")
        return

    # Create results directory
    os.makedirs(os.path.join(ROOT,'results'), exist_ok=True)
    os.makedirs(os.path.join(ROOT,'models'),  exist_ok=True)

    # Run simulation
    from v2v_controller import V2VController

    if args.novv:
        print("\n[Mode] Running NO-V2V comparison run...")
        ctrl = V2VController(no_v2v_mode=True)
    else:
        print("\n[Mode] Running FULL V2V simulation...")
        print("       → SUMO-GUI will open shortly")
        print("       → Dashboard window will also open")
        print("       → Watch all 4 scenario phases play out\n")
        ctrl = V2VController(no_v2v_mode=False)

    ctrl.run()

if __name__ == '__main__':
    main()