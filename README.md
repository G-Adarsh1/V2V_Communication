# Emergency-Focused V2V Intelligence System
**Team:** Ch Rakesh (23911A05D5) | G Adarsh (23911A05E2) | N JayVardhan (23911A05H4)
**Faculty Guide:** Mr Naqueeb Ahmed

---

## What This Does

A city intersection simulation in SUMO-GUI where:
- 18 normal vehicles drive through a 4-way intersection
- ML (Random Forest + LSTM) predicts collision risk in real time
- Vehicles change colour: **grey=safe, orange=risk, red=collision**
- **Phase 2:** A brake-failed car + wrong-way driver appear simultaneously → nearby vehicles slow down + change lane
- **Phase 3:** A pedestrian crosses → all approaching cars stop
- **Phase 4:** Ambulance + Fire Truck enter → traffic lights turn green → vehicles yield and clear a corridor
- A live dashboard shows all metrics updating in real time
- After simulation: graphs, CSV logs, and a comparison report are auto-saved

---

## STEP-BY-STEP: How to Run

### Step 1 — Install Python libraries
Open Command Prompt and run:
```
pip install numpy scikit-learn matplotlib
```

### Step 2 — Extract this ZIP
Extract to any folder, e.g. `C:\v2v_sumo_project\`

### Step 3 — Build the SUMO road network (do this ONCE)
Open Command Prompt in the project folder and run:
```
python run_simulation.py --build
```
This creates `map\intersection.net.xml`. You only need to do this once.

### Step 4 — Run the full simulation
```
python run_simulation.py
```

This will:
1. Train the ML models (takes ~15 seconds, shows AUC + R² scores)
2. Open **SUMO-GUI** with the intersection
3. Open a **live Dashboard** window alongside it
4. Run all 4 phases automatically

### Step 5 — Watch the simulation
- Press **Play** (▶) in SUMO-GUI if it pauses
- Watch vehicles change colour as risk is detected
- Watch the Dashboard for live metrics and alerts
- The simulation runs for ~90 seconds then saves results

### Step 6 — View results
After the simulation ends, open the `results\` folder:
- `risk_timeline_*.png` — risk levels over time
- `pdr_delay_*.png` — network performance
- `comparison_*.png` — V2V vs no-V2V
- `impact_*.png` — vehicles saved, emergency response time
- `vehicle_log_*.csv` — every vehicle's state every second
- `alert_log_*.csv` — all V2V alerts with hop counts
- `report_*.txt` — full text summary report

---

## Scenario Timeline

| Time | Event |
|------|-------|
| 0–24s | Normal traffic through intersection |
| 25s | **Brake failure** car appears (red) + **Wrong-way driver** (orange-red) |
| 40s | **Pedestrian** crosses → vehicles stop |
| 55s | **Ambulance** (pink) + **Fire Truck** (dark red) enter → lights go green → corridor clears |
| 90s | Simulation ends, reports generated |

---

## What the Colours Mean in SUMO-GUI

| Colour | Meaning |
|--------|---------|
| Grey/White | Normal vehicle, SAFE |
| **Orange** | RISK detected by ML |
| **Red** | COLLISION imminent |
| **Bright Red** | Brake-failed vehicle |
| **Pink** | Ambulance |
| **Dark Red** | Fire Truck |
| **Blue** | Yielding to emergency vehicle |
| **Yellow** | Stopped for pedestrian |
| **Green** | Vehicle saved by V2V alert |

---

## Adjusting Simulation Speed
In SUMO-GUI: use the **delay slider** at the top to slow down or speed up.
Recommended: set delay to **80ms** for comfortable viewing during demo.

---

## File Structure
```
v2v_sumo_project/
├── map/
│   ├── intersection.nod.xml    road nodes
│   ├── intersection.edg.xml    road edges
│   ├── intersection.con.xml    lane connections
│   ├── intersection.net.xml    compiled network (auto-built)
│   ├── routes.rou.xml          all vehicles and routes
│   └── simulation.sumocfg      SUMO config
├── src/
│   ├── ml_models.py            Random Forest + LSTM
│   ├── v2v_network.py          V2V beacons, multi-hop, RSU
│   ├── v2v_controller.py       TraCI main controller
│   ├── dashboard.py            Live Tkinter dashboard
│   └── reporter.py             Graphs + CSV + report
├── results/                    auto-created, all outputs go here
├── models/                     auto-created, saved ML models
├── run_simulation.py           SINGLE FILE TO RUN EVERYTHING
└── requirements.txt
```
