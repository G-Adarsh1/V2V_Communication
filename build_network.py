#!/usr/bin/env python3
"""
Build the SUMO network file from node/edge/connection XML files.
Run this once: python build_network.py
Requires SUMO to be installed at C:\Program Files (x86)\Eclipse\Sumo
"""
import subprocess, os, sys

SUMO_HOME = r"C:\Program Files (x86)\Eclipse\Sumo"
NETCONVERT = os.path.join(SUMO_HOME, "bin", "netconvert.exe")
MAP_DIR = os.path.join(os.path.dirname(__file__), "map")

def build():
    cmd = [
        NETCONVERT,
        "--node-files",       os.path.join(MAP_DIR, "intersection.nod.xml"),
        "--edge-files",       os.path.join(MAP_DIR, "intersection.edg.xml"),
        "--connection-files", os.path.join(MAP_DIR, "intersection.con.xml"),
        "--output-file",      os.path.join(MAP_DIR, "intersection.net.xml"),
        "--no-turnarounds",   "true",
        "--tls.default-type", "actuated",
        "--junctions.corner-detail", "5",
        "--geometry.remove",  "true",
        "--roundabouts.guess","true",
        "--sidewalks.guess",  "true",
        "--crossings.guess",  "true",
        "--walkingarea-output", os.path.join(MAP_DIR, "walking.net.xml"),
    ]
    print("[netconvert] Building network...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("[netconvert] SUCCESS — intersection.net.xml created.")
    else:
        print("[netconvert] ERROR:")
        print(result.stderr)
        sys.exit(1)

if __name__ == "__main__":
    build()