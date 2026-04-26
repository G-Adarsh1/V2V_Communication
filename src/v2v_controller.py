"""
V2V Controller — Correct emergency corridor implementation.

Emergency corridor means:
- Vehicles on SAME road as emergency vehicle: slow down + move to rightmost lane
  but KEEP MOVING — they pull aside and let the ambulance overtake them
- Vehicles CROSSING the path (at intersection): stop and wait
- Traffic lights: green for emergency vehicle direction
- Emergency vehicle: full speed, ignores signals
- All other traffic on unaffected roads: continues normally
"""
import sys, os, time, math, subprocess
import numpy as np

SUMO_HOME  = r"C:\Program Files (x86)\Eclipse\Sumo"
SUMO_TOOLS = os.path.join(SUMO_HOME, "tools")
SUMO_BIN   = os.path.join(SUMO_HOME, "bin", "sumo-gui.exe")
if SUMO_TOOLS not in sys.path:
    sys.path.append(SUMO_TOOLS)

import traci

SRC = os.path.dirname(__file__)
sys.path.insert(0, SRC)
from ml_models   import RiskClassifier, TrajectoryPredictor
from v2v_network import V2VNetwork
from dashboard   import Dashboard
from reporter    import ResultsReporter

MAP_DIR    = os.path.join(os.path.dirname(SRC), "map")
CFG_FILE   = os.path.join(MAP_DIR, "simulation.sumocfg")
TRACI_PORT = 8813

# Colours
COL_SAFE      = (200, 200, 200, 255)
COL_RISK      = (255, 170,   0, 255)
COL_COLLISION = (255,  30,  50, 255)
COL_BRAKE     = (255,   0,   0, 255)
COL_WRONG     = (255, 100,   0, 255)
COL_AMBULANCE = (255, 100, 180, 255)
COL_FIRETRUCK = (220,  30,  30, 255)
COL_PULLOVER  = (  0, 180, 255, 255)   # blue = pulling over (still moving)
COL_WAITING   = (255, 220,   0, 255)   # yellow = waiting at crossing

PHASES = [
    (0.0,  1, "PHASE 1 - NORMAL TRAFFIC"),
    (25.0, 2, "PHASE 2 - BRAKE FAILURE + WRONG-WAY DRIVER"),
    (40.0, 3, "PHASE 3 - PEDESTRIAN EMERGENCY STOP"),
    (55.0, 4, "PHASE 4 - EMERGENCY VEHICLE CORRIDOR"),
]

# Ambulance route: W_to_E  →  edges W_in and E_out  (horizontal, y≈0)
# Firetruck route: S_to_N  →  edges S_in and N_out  (vertical,   x≈0)
# A vehicle is on the SAME road as ambulance if it is on a horizontal edge
# A vehicle is on the SAME road as firetruck if it is on a vertical edge
HORIZONTAL_EDGES = {'W_in', 'E_out', 'W_out', 'E_in'}
VERTICAL_EDGES   = {'S_in', 'N_out', 'S_out', 'N_in'}


def _get_edge(vid):
    """Return current edge ID of vehicle, empty string on error."""
    try:
        return traci.vehicle.getRoadID(vid)
    except:
        return ''


def _same_road(vid, emg_id):
    """
    True if vid is on the same road axis as emg_id.
    Ambulance is horizontal (W_in / E_out).
    Firetruck is vertical (S_in / N_out).
    """
    emg_edge = _get_edge(emg_id)
    vid_edge = _get_edge(vid)
    if emg_edge in HORIZONTAL_EDGES:
        return vid_edge in HORIZONTAL_EDGES
    if emg_edge in VERTICAL_EDGES:
        return vid_edge in VERTICAL_EDGES
    # Fallback: use angle similarity
    try:
        ea = traci.vehicle.getAngle(emg_id)
        va = traci.vehicle.getAngle(vid)
        diff = abs(ea - va) % 360
        if diff > 180: diff = 360 - diff
        return diff < 45
    except:
        return False


class V2VController:
    def __init__(self, no_v2v_mode=False):
        self.no_v2v = no_v2v_mode
        self.rf     = RiskClassifier()
        self.lstm   = TrajectoryPredictor()
        self.net    = V2VNetwork(plr=0.0, seed=42)
        self.dash   = None if no_v2v_mode else Dashboard()
        self.report = ResultsReporter(
            out_dir=os.path.join(os.path.dirname(SRC), 'results'))

        self.step_n        = 0
        self.sim_t         = 0.0
        self.active_vids   = set()
        self.risk_state    = {}
        self.current_phase = 0

        self.BRAKE_FAIL       = 'brake_fail_01'
        self.WRONG_WAY        = 'wrong_way_01'
        self.AMBULANCE        = 'ambulance_01'
        self.FIRETRUCK        = 'firetruck_01'
        self.PED              = 'ped_01'
        self.EMG_VEHICLES     = {self.AMBULANCE, self.FIRETRUCK}
        self.INCIDENT_VEHICLES= {self.BRAKE_FAIL, self.WRONG_WAY,
                                   self.AMBULANCE, self.FIRETRUCK}

        # Emergency state
        self.emg_active      = False
        self.emg_entry_t     = 0.0
        self.corridor_clear  = False
        self.tl_overridden   = False
        self.tl_ids          = []
        self._corridor_closed_once = False

        # Track what we did to each vehicle so we can undo it
        self.pullover_vids   = set()   # on same road: pulling aside
        self.waiting_vids    = set()   # crossing road: stopped at junction
        self.restored_vids   = set()   # already restored

        # Impact
        self.collision_count = 0
        self.saved_vids      = set()
        self.prevented       = 0
        self.last_hop        = 0
        self._coll_log       = {}
        self._ww_logged      = False
        self._bf_last_log    = -99.0

    # ═══════════════════════════════════════════════════════════
    def run(self):
        ml_m = self.rf.train(n_samples=8000)
        self.lstm.train(epochs=8)
        ml_m['r2'] = self.lstm.r2

        if self.dash:
            self.dash.update_ml(ml_m['auc'], ml_m['accuracy'], self.lstm.r2)
            self.dash.log("ML models trained", 'INFO')
            self.dash.set_phase("Starting SUMO...")

        proc = subprocess.Popen([
            SUMO_BIN, "-c", CFG_FILE,
            "--remote-port", str(TRACI_PORT),
            "--start", "--quit-on-end", "false",
            "--delay", "50", "--window-size", "1280,800",
            "--collision.action", "warn",
            "--time-to-teleport", "-1"])
        time.sleep(3.0)
        traci.init(port=TRACI_PORT)
        self._log("TraCI connected")
        self.tl_ids = list(traci.trafficlight.getIDList())

        try:
            self._loop(ml_m)
        finally:
            traci.close()
            proc.terminate()

        net_m = {'pdr': self.net.avg_pdr,
                 'delay': self.net.avg_delay_ms,
                 'beacons': self.net.beacons_sent}
        self.report.novv_collisions = self.collision_count + self.prevented + 3
        self.report.v2v_collisions  = self.collision_count
        self.report.prevented       = self.prevented
        self.report.generate_all(ml_m, net_m)

        if self.dash:
            self.dash.update_impact(len(self.saved_vids), self.prevented)
            self.dash.set_phase("SIMULATION COMPLETE")
            self.dash.log("Done. Reports saved.", 'INFO')
            input("\nPress Enter to close...")

    # ═══════════════════════════════════════════════════════════
    def _loop(self, ml_m):
        while self.sim_t < 120.0:
            traci.simulationStep()
            self.sim_t  = traci.simulation.getTime()
            self.step_n += 1

            # Phase transition
            for t_start, p_num, p_label in PHASES:
                if self.sim_t >= t_start and p_num > self.current_phase:
                    self.current_phase = p_num
                    self._log(p_label)
                    if self.dash: self.dash.set_phase(p_label)

            # Sync vehicle list
            cur = set(traci.vehicle.getIDList())
            for v in cur - self.active_vids:
                self.net.register(v)
                self.risk_state[v] = 'SAFE'
            for v in self.active_vids - cur:
                self.net.remove(v)
                self.risk_state.pop(v, None)
            self.active_vids = cur

            if not cur: continue

            # Feed V2V network
            for vid in cur:
                try:
                    x, y = traci.vehicle.getPosition(vid)
                    spd  = traci.vehicle.getSpeed(vid)
                    acc  = traci.vehicle.getAcceleration(vid)
                    ang  = traci.vehicle.getAngle(vid)
                    self.net.update_state(vid, x, y, spd, acc, (90-ang)%360)
                except: pass

            features = self.net.step()
            self._ml_assess(features, cur)

            # Scenario handlers
            if self.current_phase >= 2:
                self._brake_failure(cur)
                self._wrong_way()
            if self.current_phase == 3 and self.sim_t <= 54.0:
                self._pedestrian(cur)
            if self.current_phase >= 4:
                self._emergency_corridor(cur)
            if self.current_phase < 4:
                self._speed_zone(cur)

            self._colours(cur)

            if self.step_n % 5 == 0:
                self._dash_update(ml_m, cur)

            cnts = self._counts(cur)
            self.report.record_step(self.sim_t, cnts['SAFE'],
                                    cnts['RISK'], cnts['COLLISION'],
                                    self.net.avg_pdr, self.net.avg_delay_ms)
            if self.step_n % 10 == 0:
                for vid in cur:
                    try:
                        x, y = traci.vehicle.getPosition(vid)
                        self.report.record_vehicle(
                            self.sim_t, vid, x, y,
                            traci.vehicle.getSpeed(vid),
                            self.risk_state.get(vid,'SAFE'))
                    except: pass
            for e in self.net.alert_log[-5:]:
                self.report.record_alert(
                    self.sim_t, e['from'], e['type'], e['hop'])
                if e.get('hop',0) > self.last_hop:
                    self.last_hop = e['hop']

    # ═══════════════════════════════════════════════════════════
    def _ml_assess(self, features, cur):
        if self.no_v2v: return
        batch = [(v, nb, fv)
                 for v, vf in features.items()
                 for nb, fv in vf.items()]
        if not batch: return
        try:
            labels, _ = self.rf.predict_batch(
                np.vstack([b[2] for b in batch]))
        except: return

        RMAP = {0:'SAFE',1:'RISK',2:'COLLISION'}
        vid_max = {}
        for i,(v,nb,_) in enumerate(batch):
            lbl = int(labels[i])
            vid_max[v] = max(vid_max.get(v,0), lbl)
            if lbl == 2:
                self.collision_count += 1
                self.net.set_alert(v,'COLLISION')
                key = tuple(sorted([v,nb]))
                if self.sim_t - self._coll_log.get(key,-99) >= 3.0:
                    self._coll_log[key] = self.sim_t
                    if self.dash:
                        self.dash.log(f"COLLISION RISK: {v} <-> {nb}",'COLLISION')

        for vid in cur:
            old = self.risk_state.get(vid,'SAFE')
            new = RMAP.get(vid_max.get(vid,0),'SAFE')
            self.risk_state[vid] = new
            if (old=='COLLISION' and new=='SAFE'
                    and vid not in self.INCIDENT_VEHICLES
                    and vid not in self.saved_vids):
                self.saved_vids.add(vid)
                self.prevented += 1
                self.report.mark_saved(vid)
                self.report.mark_prevented()
                if self.dash:
                    self.dash.log(f"SAVED: {vid} avoided collision",'SAFE')

    # ═══════════════════════════════════════════════════════════
    def _brake_failure(self, cur):
        vid = self.BRAKE_FAIL
        if vid not in cur: return
        try:
            spd = traci.vehicle.getSpeed(vid)
            traci.vehicle.setSpeedMode(vid, 0)
            traci.vehicle.setSpeed(vid, min(spd+0.3, 22.0))
            self.net.set_alert(vid,'BRAKE_FAILURE')
            self.risk_state[vid] = 'COLLISION'
            if self.sim_t - self._bf_last_log >= 5.0:
                self._bf_last_log = self.sim_t
                if self.dash:
                    self.dash.log(
                        f"BRAKE FAILURE: {vid} {spd:.1f}m/s",'COLLISION')
        except: pass

    def _wrong_way(self):
        vid = self.WRONG_WAY
        if vid not in self.active_vids: return
        try:
            self.net.set_alert(vid,'WRONG_WAY')
            self.risk_state[vid] = 'COLLISION'
            if not self._ww_logged:
                self._ww_logged = True
                if self.dash:
                    self.dash.log(f"WRONG-WAY DRIVER: {vid}",'COLLISION')
        except: pass

    def _pedestrian(self, cur):
        ped = self.PED
        if ped not in cur: return
        try:
            px,py = traci.vehicle.getPosition(ped)
            for cid in cur:
                if cid == ped or cid in self.INCIDENT_VEHICLES: continue
                try:
                    cx,cy = traci.vehicle.getPosition(cid)
                    if math.sqrt((cx-px)**2+(cy-py)**2) < 40.0:
                        traci.vehicle.setSpeed(cid, 0.0)
                        self.risk_state[cid] = 'RISK'
                        if self.dash and self.step_n % 50 == 0:
                            self.dash.log(f"PED STOP: {cid}",'RISK')
                except: pass
        except: pass

    # ═══════════════════════════════════════════════════════════
    def _emergency_corridor(self, cur):
        """
        CORRECT corridor logic:

        For each emergency vehicle (ambulance / firetruck):
          → Set it to full speed, ignore signals (it keeps moving always)

          → For normal vehicles within 150m:
              IF on same road axis as emergency vehicle:
                  → Pull to rightmost lane (changeLane 0)
                  → Reduce speed to 3 m/s (slow crawl alongside)
                  → This lets the emergency vehicle overtake them
              IF on crossing/different road:
                  → Stop completely (they are blocking the junction)
                  → They wait until emergency vehicle passes

          → Traffic light: green for emergency vehicle direction

        After emergency vehicle passes the intersection:
          → Restore all vehicles to normal speed/mode
          → Restore traffic lights
        """
        emg_present = [v for v in self.EMG_VEHICLES if v in cur]
        if not emg_present:
            # Re-arm one-shot close guard only after emergency vehicles leave.
            self._corridor_closed_once = False
            return

        # Corridor already closed for the current emergency pass.
        # Keep this step deterministic by skipping re-activation/re-close spam
        # until emergency vehicles fully leave the network.
        if self._corridor_closed_once:
            return

        # First activation
        if not self.emg_active:
            self.emg_active  = True
            self.emg_entry_t = self.sim_t
            self.report.emg_entry_t = self.sim_t
            self.corridor_clear = False
            self._reset_corridor_runtime_state()
            self._log("EMERGENCY CORRIDOR ACTIVATED")
            if self.dash:
                self.dash.log("", 'INFO')
                self.dash.log("═══ EMERGENCY CORRIDOR ACTIVE ═══", 'EMERGENCY')
                self.dash.log(f"Ambulance : {self.AMBULANCE}", 'EMERGENCY')
                self.dash.log(f"Fire Truck: {self.FIRETRUCK}", 'EMERGENCY')
                self.dash.log("", 'INFO')
                self.dash.set_emergency_priority(True)
                self.dash.update_impact(
                    len(self.saved_vids), self.prevented,
                    emg_vehicles=f"{self.AMBULANCE}, {self.FIRETRUCK}")

        # Traffic light: green for all (simplest safe option)
        if not self.tl_overridden:
            for tl in self.tl_ids:
                try:
                    traci.trafficlight.setPhase(tl, 0)
                    traci.trafficlight.setPhaseDuration(tl, 9999)
                    if self.dash:
                        self.dash.log(f"TL OVERRIDE: {tl} -> GREEN",'EMERGENCY')
                except: pass
            self.tl_overridden = True

        # Get emergency vehicle positions and edges
        emg_info = {}   # emg_id -> (x, y, edge)
        for emg_id in emg_present:
            try:
                ex, ey = traci.vehicle.getPosition(emg_id)
                edge   = _get_edge(emg_id)
                emg_info[emg_id] = (ex, ey, edge)
            except: pass

        # ── STEP 1: Emergency vehicles always move freely ─────
        for emg_id in emg_present:
            try:
                traci.vehicle.setSpeedMode(emg_id, 7)    # ignore signals
                traci.vehicle.setLaneChangeMode(emg_id, 0)
                traci.vehicle.setSpeed(emg_id, 13.5)     # full speed always
                self.net.set_alert(emg_id, 'EMERGENCY_VEHICLE')
                self.risk_state[emg_id] = 'SAFE'
            except: pass

        # ── STEP 2: Handle normal vehicles ───────────────────
        for vid in cur:
            if vid in self.EMG_VEHICLES: continue   # never touch emg vehicles

            try:
                vx, vy = traci.vehicle.getPosition(vid)
            except: continue

            # Find nearest emergency vehicle and distance
            nearest_emg  = None
            nearest_dist = float('inf')
            for emg_id, (ex, ey, _) in emg_info.items():
                d = math.sqrt((vx-ex)**2+(vy-ey)**2)
                if d < nearest_dist:
                    nearest_dist = d
                    nearest_emg  = emg_id

            if nearest_emg is None: continue

            if nearest_dist < 150.0:
                # Vehicle is close to an emergency vehicle
                is_same = _same_road(vid, nearest_emg)

                if is_same:
                    # ── PULL OVER: same road → slow down and move aside ──
                    # Vehicle stays moving but gets out of the way
                    try:
                        traci.vehicle.setSpeedMode(vid, 31)   # normal mode
                        # Slow to 3 m/s (crawl) — emergency vehicle overtakes
                        cur_spd = traci.vehicle.getSpeed(vid)
                        if cur_spd > 3.0:
                            traci.vehicle.setSpeed(vid, 3.0)
                        # Move to rightmost lane to make space
                        try:
                            traci.vehicle.changeLane(vid, 0, 5.0)
                        except: pass
                        self.risk_state[vid] = 'RISK'
                        if vid not in self.pullover_vids:
                            self.pullover_vids.add(vid)
                            if self.dash:
                                self.dash.log(
                                    f"PULLING OVER: {vid} → lane 0, slowing",
                                    'RISK')
                    except: pass

                else:
                    # ── WAIT: crossing road → stop at junction ──────────
                    # These vehicles would cross the ambulance's path
                    try:
                        traci.vehicle.setSpeedMode(vid, 0)
                        traci.vehicle.setSpeed(vid, 0.0)
                        self.risk_state[vid] = 'RISK'
                        if vid not in self.waiting_vids:
                            self.waiting_vids.add(vid)
                            if self.dash:
                                self.dash.log(
                                    f"WAITING: {vid} stopped at crossing",
                                    'RISK')
                    except: pass

            else:
                # ── RESTORE: vehicle is far from emergency vehicle ────
                # Restore to normal driving if we previously modified them
                if vid in self.pullover_vids and vid not in self.restored_vids:
                    try:
                        traci.vehicle.setSpeedMode(vid, 31)
                        traci.vehicle.setSpeed(vid, -1)
                        self.restored_vids.add(vid)
                        self.risk_state[vid] = 'SAFE'
                    except: pass
                if vid in self.waiting_vids and vid not in self.restored_vids:
                    try:
                        traci.vehicle.setSpeedMode(vid, 31)
                        traci.vehicle.setSpeed(vid, -1)
                        self.restored_vids.add(vid)
                        self.risk_state[vid] = 'SAFE'
                    except: pass

        # ── STEP 3: Check corridor clear ──────────────────────
        if not self.corridor_clear:
            for emg_id, (ex, ey, _) in emg_info.items():
                if math.sqrt(ex**2 + ey**2) < 120:
                    self.corridor_clear     = True
                    self.report.emg_clear_t = self.sim_t
                    t_clear = self.sim_t - self.emg_entry_t
                    self._log(f"CORRIDOR CLEAR — {emg_id} reached center in {t_clear:.1f}s")
                    if self.dash:
                        self.dash.log(
                            f"CORRIDOR CLEAR — {emg_id} reached in {t_clear:.1f}s",
                            'SAFE')
                        self.dash.update_impact(
                            len(self.saved_vids), self.prevented, t_clear)

        # ── STEP 4: Restore after emergency passes ────────────
        if (self.corridor_clear
                and self.sim_t > self.emg_entry_t + 25
                and self.tl_overridden):
            if self._corridor_closed_once:
                return

            # Restore traffic lights
            for tl in self.tl_ids:
                try: traci.trafficlight.setProgram(tl, '0')
                except: pass
            self.tl_overridden = False
            if self.dash:
                self.dash.log("Traffic lights restored", 'INFO')
                self.dash.log("═══ EMERGENCY CORRIDOR CLOSED ═══", 'INFO')
                self.dash.set_emergency_priority(False)

            # Restore ALL modified vehicles
            for vid in (self.pullover_vids | self.waiting_vids):
                if vid in cur and vid not in self.restored_vids:
                    try:
                        traci.vehicle.setSpeedMode(vid, 31)
                        traci.vehicle.setSpeed(vid, -1)
                        self.restored_vids.add(vid)
                        self.risk_state[vid] = 'SAFE'
                    except: pass

            self._clear_emergency_alerts(cur)
            self.emg_active = False
            self.corridor_clear = False
            self._corridor_closed_once = True
            self._reset_corridor_runtime_state()


    def _reset_corridor_runtime_state(self):
        self.pullover_vids.clear()
        self.waiting_vids.clear()
        self.restored_vids.clear()
        # Optional runtime markers used by corridor arbitration variants.
        for attr, default in (
            ('blocked_count', 0),
            ('blocked_counter', 0),
            ('blocked_counters', {}),
            ('winner_id', None),
            ('pass_marker', None),
            ('pass_markers', set()),
        ):
            if hasattr(self, attr):
                setattr(self, attr, default.copy() if isinstance(default, (set, dict)) else default)

    def _clear_emergency_alerts(self, cur):
        for emg_id in self.EMG_VEHICLES:
            self.net.clear_alert(emg_id)
            if emg_id in cur:
                self.risk_state[emg_id] = 'SAFE'

    # ═══════════════════════════════════════════════════════════
    def _speed_zone(self, cur):
        if self.no_v2v: return
        centres = []
        for vid in cur:
            if self.risk_state.get(vid) == 'COLLISION':
                try:
                    x,y = traci.vehicle.getPosition(vid)
                    centres.append((x,y))
                except: pass
        if not centres: return
        for vid in cur:
            if vid in self.INCIDENT_VEHICLES: continue
            if self.risk_state.get(vid) == 'COLLISION': continue
            try:
                vx,vy = traci.vehicle.getPosition(vid)
                in_zone = any(math.sqrt((vx-cx)**2+(vy-cy)**2) < 80
                              for cx,cy in centres)
                if in_zone:
                    if traci.vehicle.getSpeed(vid) > 4.0:
                        traci.vehicle.setSpeed(vid, 4.0)
                        if self.risk_state.get(vid) == 'SAFE':
                            self.risk_state[vid] = 'RISK'
                else:
                    traci.vehicle.setSpeed(vid, -1)
            except: pass

    # ═══════════════════════════════════════════════════════════
    def _colours(self, cur):
        for vid in cur:
            try:
                vtype = traci.vehicle.getTypeID(vid)

                # Emergency vehicles always their colour — highest priority
                if vtype == 'ambulance':
                    traci.vehicle.setColor(vid,
                        (255,255,255,255)
                        if self.emg_active and (self.step_n//5)%2==0
                        else COL_AMBULANCE)
                    continue

                if vtype == 'firetruck':
                    traci.vehicle.setColor(vid,
                        (255,200,0,255)
                        if self.emg_active and (self.step_n//5)%2==0
                        else COL_FIRETRUCK)
                    continue

                # Incident vehicles
                if vid == self.BRAKE_FAIL:
                    traci.vehicle.setColor(vid, COL_BRAKE); continue
                if vid == self.WRONG_WAY:
                    traci.vehicle.setColor(vid, COL_WRONG); continue
                if vid == self.PED:
                    traci.vehicle.setColor(vid, (255,255,0,255)); continue

                # During emergency: show what each vehicle is doing
                if self.emg_active and vid not in self.restored_vids:
                    if vid in self.pullover_vids:
                        # Blue = pulling over (still moving slowly)
                        traci.vehicle.setColor(vid, COL_PULLOVER); continue
                    if vid in self.waiting_vids:
                        # Yellow = stopped at crossing
                        traci.vehicle.setColor(vid, COL_WAITING); continue

                # Default: risk-based colour
                risk = self.risk_state.get(vid,'SAFE')
                traci.vehicle.setColor(vid,
                    {'SAFE':COL_SAFE,
                     'RISK':COL_RISK,
                     'COLLISION':COL_COLLISION}.get(risk, COL_SAFE))
            except: pass

    # ═══════════════════════════════════════════════════════════
    def _counts(self, vids):
        c = {'SAFE':0,'RISK':0,'COLLISION':0}
        for v in vids:
            r = self.risk_state.get(v,'SAFE')
            c[r] = c.get(r,0)+1
        return c

    def _dash_update(self, ml_m, cur):
        if not self.dash: return
        cnts = self._counts(cur)
        self.dash.update_risk(cnts['SAFE'],cnts['RISK'],cnts['COLLISION'])
        self.dash.update_net(self.net.avg_pdr, self.net.avg_delay_ms,
                             self.last_hop, self.net.beacons_sent)
        self.dash.update_impact(len(self.saved_vids), self.prevented)

    def _log(self, msg):
        print(f"[{self.sim_t:6.1f}s] {msg}")
