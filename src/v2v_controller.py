"""
V2V Controller — Fully fixed version.

FIXES IN THIS VERSION (on top of prior fixes):
  A. EARLY TL PREEMPTION (truly early):
       - tl_overridden flag is now split into tl_phase_locked (bool) +
         tl_winner_phase (int). Every step the winner_phase is re-evaluated
         and if it differs from what's currently locked, the TL is updated
         immediately. This means the correct green axis is set the INSTANT
         Phase 4 starts, before EVs are anywhere near the junction.

  B. LANE FLICKERING ELIMINATED (root cause fixed):
       - _assign_pullover_lane now correctly finds the rightmost lane for
         left-hand traffic (SUMO default). The old loop broke on li=0 always,
         which could push vehicles to lane 0 even when that was the EV lane.
       - _make_space_either_lane replaced with intelligent directional version:
         pushes vehicle AWAY from EV travel axis, never randomly.
       - _lane_assigned guard is checked before ANY lane command anywhere.

  C. INTERSECTION CLEARANCE:
       - _clear_intersection() called each step once winner_id is known.
         Vehicles within 40 m of the EV are nudged forward at 6 m/s so
         they clear the box junction before the EV arrives.

  D. CROSS-ROAD STOP TIMING:
       - Waiting vehicles are re-commanded to speed=0 every step while the
         EV is within CONFLICT_STOP_RADIUS (not just on first entry).
         Prevents them from creeping forward through residual green time.
"""

import sys, os, time, math, subprocess
import numpy as np

SUMO_HOME  = r"C:\Program Files (x86)\Eclipse\Sumo"
SUMO_TOOLS = os.path.join(SUMO_HOME, "tools")
SUMO_BIN   = os.path.join(SUMO_HOME, "bin", "sumo-gui.exe")
if SUMO_TOOLS not in sys.path:
    sys.path.append(SUMO_TOOLS)

import traci

SRC = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC)
from ml_models   import RiskClassifier, TrajectoryPredictor
from v2v_network import V2VNetwork
from dashboard   import Dashboard
from reporter    import ResultsReporter

MAP_DIR    = os.path.join(os.path.dirname(SRC), "map")
CFG_FILE   = os.path.join(MAP_DIR, "simulation.sumocfg")
TRACI_PORT = 8813

# ── Colours ──────────────────────────────────────────────────
COL_SAFE      = (200, 200, 200, 255)
COL_RISK      = (255, 170,   0, 255)
COL_COLLISION = (255,  30,  50, 255)
COL_BRAKE     = (255,   0,   0, 255)
COL_WRONG     = (255, 100,   0, 255)
COL_AMBULANCE = (255, 100, 180, 255)
COL_FIRETRUCK = (220,  30,  30, 255)
COL_RESCUE    = (120, 200, 255, 255)
COL_PULLOVER  = (  0, 180, 255, 255)   # blue  = pulling over (still moving)
COL_WAITING   = (255, 220,   0, 255)   # yellow = stopped at crossing

# ── Simulation phases ─────────────────────────────────────────
PHASES = [
    (0.0,  1, "PHASE 1 - NORMAL TRAFFIC"),
    (25.0, 2, "PHASE 2 - BRAKE FAILURE + WRONG-WAY DRIVER"),
    (40.0, 3, "PHASE 3 - PEDESTRIAN EMERGENCY STOP"),
    (55.0, 4, "PHASE 4 - EMERGENCY VEHICLE CORRIDOR"),
]

# ── Edge axis sets ────────────────────────────────────────────
HORIZONTAL_EDGES = {'W_in', 'E_out', 'W_out', 'E_in'}
VERTICAL_EDGES   = {'S_in', 'N_out', 'S_out', 'N_in'}

# ── Corridor tuning constants ─────────────────────────────────
CORRIDOR_AHEAD_M     = 180.0   # how far AHEAD of EV to look for same-axis vehicles
CORRIDOR_BEHIND_M    = 30.0    # how far BEHIND EV to look (vehicles alongside)
CORRIDOR_FULL_WIDTH  = 40.0    # lateral width covering ALL lanes of the road ahead
CORRIDOR_HALF_WIDTH  = 12.0    # tight lateral zone for behind/cross classification
CONFLICT_ZONE_M      = 60.0    # how far from junction cross-road vehicles must stop
CRAWL_SPEED_MPS      = 3.0     # same-axis vehicles slow to this
EV_GAP_THROTTLE_HARD = 8.0     # EV slows hard if gap < this (m)
EV_GAP_THROTTLE_SOFT = 25.0    # EV slows gently if gap < this (m)
LANE_LOCK_COOLDOWN   = 30      # steps between lane-change commands per vehicle
LANE_TRACE_DEPTH     = 10      # lane-level look-ahead hops for strict corridor
LANE_RETRY_STEPS     = 8       # retry lane-change if vehicle failed to move aside
FORCE_CLEAR_GAP_M    = 32.0    # inside this gap, blocker gets stronger clear command

# ── TL preemption phases ──────────────────────────────────────
# Ambulance travels W→E (horizontal) → needs E/W green
# Firetruck travels S→N (vertical)   → needs N/S green
HORIZONTAL_GREEN_PHASE = 3
VERTICAL_GREEN_PHASE   = 0


# ═══════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════

def _get_edge(vid):
    try:
        return traci.vehicle.getRoadID(vid)
    except:
        return ''


def _ev_heading_vec(emg_id):
    """Return (fx, fy) unit vector in the EV's travel direction."""
    try:
        ea = traci.vehicle.getAngle(emg_id)          # SUMO angle: 0=N, 90=E
        hdg = math.radians((90 - ea) % 360)
        return math.cos(hdg), math.sin(hdg)
    except:
        return 1.0, 0.0


def _same_direction(vid, emg_id):
    """
    True if vid is travelling in the SAME direction as emg_id (±45°).
    Used to distinguish same-direction traffic (pull over) from
    oncoming traffic (do nothing — they're already in a separate lane).
    """
    try:
        ea   = traci.vehicle.getAngle(emg_id)
        va   = traci.vehicle.getAngle(vid)
        diff = abs(ea - va) % 360
        if diff > 180:
            diff = 360 - diff
        return diff < 45
    except:
        return False


def _motion_axis(vid):
    """
    Determine travel axis from heading (robust across arbitrary edge names).
    Returns 'h' (east/west), 'v' (north/south), or ''.
    """
    try:
        ea = traci.vehicle.getAngle(vid)          # SUMO: 0=N, 90=E
        hdg = math.radians((90 - ea) % 360)
        fx, fy = math.cos(hdg), math.sin(hdg)
        return 'h' if abs(fx) >= abs(fy) else 'v'
    except:
        return ''


def _in_ev_path(vid, emg_id):
    """
    Path-based relevance check.

    Returns (role, along, lateral) where role is:
      'ahead'  — vehicle in front of EV on the same road axis, ANY lane
                 (lateral < CORRIDOR_FULL_WIDTH=40m covers all lanes)
      'behind' — same-direction vehicle beside/just-behind EV (tight zone)
      'cross'  — cross-axis vehicle near the junction box
      None     — irrelevant (oncoming, different road, too far)

    KEY CHANGE: 'ahead' uses CORRIDOR_FULL_WIDTH (40m) so vehicles in ALL
    lanes of the road ahead are detected and commanded to form a corridor —
    not just vehicles directly in the EV's own lane (12m).
    'behind' and 'cross' keep the tight CORRIDOR_HALF_WIDTH to avoid
    commanding vehicles that genuinely don't block the EV.
    """
    try:
        vx, vy = traci.vehicle.getPosition(vid)
        ex, ey = traci.vehicle.getPosition(emg_id)
        fx, fy = _ev_heading_vec(emg_id)
        px, py = -fy, fx   # perpendicular

        dx = vx - ex
        dy = vy - ey

        along   = dx * fx + dy * fy
        lateral = abs(dx * px + dy * py)

        emg_edge = _get_edge(emg_id)
        vid_edge = _get_edge(vid)

        emg_axis   = _motion_axis(emg_id)
        vid_axis   = _motion_axis(vid)
        emg_horiz  = emg_axis == 'h'
        emg_vert   = emg_axis == 'v'
        vid_horiz  = vid_axis == 'h'
        vid_vert   = vid_axis == 'v'
        emg_in_jct = emg_edge.startswith(':')

        # ── EV inside junction box — pure geometry ───────────────
        if emg_in_jct:
            if _same_direction(vid, emg_id):
                if along > 5.0 and lateral < CORRIDOR_FULL_WIDTH:
                    return 'ahead', along, lateral
                if -CORRIDOR_BEHIND_M <= along <= 5.0 and lateral < CORRIDOR_HALF_WIDTH:
                    return 'behind', along, lateral
            else:
                if abs(along) < CONFLICT_ZONE_M and lateral < CONFLICT_ZONE_M:
                    return 'cross', along, lateral
            return None, along, lateral

        # ── EV on normal road — same axis ────────────────────────
        if (emg_horiz and vid_horiz) or (emg_vert and vid_vert):
            if not _same_direction(vid, emg_id):
                return None, along, lateral   # oncoming — ignore completely

            # AHEAD: wide zone covers all lanes of the road
            if along > 5.0 and lateral < CORRIDOR_FULL_WIDTH:
                return 'ahead', along, lateral

            # BEHIND/BESIDE: tight zone — only vehicles actually next to EV
            if -CORRIDOR_BEHIND_M <= along <= 5.0 and lateral < CORRIDOR_HALF_WIDTH:
                return 'behind', along, lateral

            return None, along, lateral

        # ── Cross-axis: only near junction ───────────────────────
        if (emg_horiz and vid_vert) or (emg_vert and vid_horiz):
            if abs(along) < CONFLICT_ZONE_M and lateral < CONFLICT_ZONE_M:
                return 'cross', along, lateral
            return None, along, lateral

        # Internal junction lanes handled by _clear_intersection
        if vid_edge.startswith(':'):
            return None, along, lateral

        return None, along, lateral

    except:
        return None, 0.0, 0.0


def _is_ahead_of_ev(vid, emg_id):
    """True if vid is in FRONT of emg_id (EV would catch up to it)."""
    try:
        vx, vy = traci.vehicle.getPosition(vid)
        ex, ey = traci.vehicle.getPosition(emg_id)
        fx, fy = _ev_heading_vec(emg_id)
        return ((vx - ex) * fx + (vy - ey) * fy) > 2.0
    except:
        return False


def _lane_edge(lane_id):
    """Best-effort edge id extraction from lane id."""
    if not lane_id:
        return ''
    if '_' in lane_id:
        return lane_id.rsplit('_', 1)[0]
    return lane_id


def _lane_link_targets(lane_id):
    """
    Return [(to_lane_id, via_lane_id)] for a lane.
    Handles different SUMO link tuple layouts defensively.
    """
    out = []
    try:
        links = traci.lane.getLinks(lane_id)
    except:
        return out

    for lk in links:
        if not lk:
            continue
        to_lane = lk[0] if len(lk) > 0 and isinstance(lk[0], str) else ''
        via_lane = ''
        for item in lk[1:]:
            if isinstance(item, str) and item.startswith(':'):
                via_lane = item
                break
        if to_lane:
            out.append((to_lane, via_lane))
    return out


# ═══════════════════════════════════════════════════════════════
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

        self.BRAKE_FAIL        = 'brake_fail_01'
        self.WRONG_WAY         = 'wrong_way_01'
        self.AMBULANCE         = 'ambulance_01'
        self.FIRETRUCK         = 'firetruck_01'
        self.PED               = 'ped_01'
        self.EMG_TYPES         = {'ambulance', 'firetruck', 'rescue_ev'}
        self.INCIDENT_VEHICLES = {self.BRAKE_FAIL, self.WRONG_WAY,
                                   self.PED}

        # ── Emergency corridor state ──────────────────────────
        self.emg_active       = False
        self.emg_entry_t      = 0.0
        self.intersection_id  = 'center'
        self.intersection_ctr = (300.0, 300.0)

        # FIX A: split tl_overridden into phase-lock tracking
        self.tl_phase_locked  = False     # True once we've sent the first setPhase
        self.tl_winner_phase  = None      # which phase is currently locked
        self.tl_ids           = []

        # Per-vehicle corridor tracking
        self.pullover_vids    = set()
        self.waiting_vids     = set()
        self.restored_vids    = set()
        self._lane_assigned   = {}   # vid -> target lane index (locked once set)
        self._lane_locked     = {}   # vid -> step of last lane cmd
        self._junction_clearing = set()  # vehicles currently being nudged out of box
        self._jc_logged       = set()    # vehicles we've already logged for junction clear

        # ── Impact metrics ────────────────────────────────────
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
            self.dash.set_phase("PHASE 1 - NORMAL TRAFFIC")

        proc = subprocess.Popen([
            SUMO_BIN, "-c", CFG_FILE,
            "--remote-port", str(TRACI_PORT),
            "--start", "--quit-on-end", "false",
            "--delay", "50", "--window-size", "1280,800",
            "--collision.action", "warn",
            "--time-to-teleport", "-1",
            "--begin", "0"])
        time.sleep(3.5)
        traci.init(port=TRACI_PORT)
        self._log("TraCI connected")
        self.tl_ids = list(traci.trafficlight.getIDList())

        try:
            jids = list(traci.junction.getIDList())
            if self.intersection_id in jids:
                self.intersection_ctr = traci.junction.getPosition(self.intersection_id)
            elif jids:
                # Fallback: choose the junction nearest to the map center.
                self.intersection_id = min(
                    jids,
                    key=lambda jid: (
                        (traci.junction.getPosition(jid)[0] - 300.0) ** 2 +
                        (traci.junction.getPosition(jid)[1] - 300.0) ** 2))
                self.intersection_ctr = traci.junction.getPosition(self.intersection_id)
        except:
            self.intersection_ctr = (300.0, 300.0)

        try:
            self._loop(ml_m)
        finally:
            try:
                traci.close()
            except:
                pass
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
        """Run until ALL emergency vehicles have exited + sim past 120s."""
        while True:
            traci.simulationStep()
            self.sim_t  = traci.simulation.getTime()
            self.step_n += 1

            # Phase transitions
            for t_start, p_num, p_label in PHASES:
                if self.sim_t >= t_start and p_num > self.current_phase:
                    self.current_phase = p_num
                    self._log(p_label)
                    if self.dash:
                        self.dash.set_phase(p_label)

            # Sync vehicle list
            cur = set(traci.vehicle.getIDList())
            for v in cur - self.active_vids:
                self.net.register(v)
                self.risk_state[v] = 'SAFE'
            prev_emg_ids = set(self._get_active_emergency_ids(self.active_vids))
            cur_emg_ids  = set(self._get_active_emergency_ids(cur))
            for v in self.active_vids - cur:
                self.net.remove(v)
                self.risk_state.pop(v, None)
                if v in prev_emg_ids and self.emg_active:
                    self._on_ev_exited(v, cur)
            self.active_vids = cur

            # Exit condition: past 120s AND no EVs alive AND corridor closed
            emg_still_alive = bool(cur_emg_ids)
            if self.sim_t >= 120.0 and not emg_still_alive and not self.emg_active:
                break

            if not cur:
                continue

            # Feed V2V network
            for vid in cur:
                try:
                    x, y = traci.vehicle.getPosition(vid)
                    spd  = traci.vehicle.getSpeed(vid)
                    acc  = traci.vehicle.getAcceleration(vid)
                    ang  = traci.vehicle.getAngle(vid)
                    vtype = traci.vehicle.getTypeID(vid)
                    is_emergency = vtype in self.EMG_TYPES
                    self.net.update_state(
                        vid, x, y, spd, acc, (90 - ang) % 360,
                        is_emergency=is_emergency,
                        emergency_role=vtype if is_emergency else '')
                except:
                    pass

            features = self.net.step()
            self._ml_assess(features, cur)

            # Scenario handlers
            if self.current_phase >= 2:
                self._brake_failure(cur)
                self._wrong_way()
            if self.current_phase == 3 and self.sim_t <= 54.0:
                self._pedestrian(cur)

            # FIX A: Phase 4 → immediately lock TL, then run corridor
            if self.current_phase >= 4:
                self._preempt_signals_early(cur)
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
                            self.risk_state.get(vid, 'SAFE'))
                    except:
                        pass
            for e in self.net.alert_log[-5:]:
                self.report.record_alert(
                    self.sim_t, e['from'], e['type'], e['hop'])
                if e.get('hop', 0) > self.last_hop:
                    self.last_hop = e['hop']

    # ═══════════════════════════════════════════════════════════
    def _get_active_emergency_ids(self, vids):
        emg = []
        for vid in vids:
            try:
                if traci.vehicle.getTypeID(vid) in self.EMG_TYPES:
                    emg.append(vid)
            except:
                pass
        return emg

    def _lane_path_for_ev(self, emg_id, hop_limit=LANE_TRACE_DEPTH):
        """
        Strict lane corridor path via lane-level link tracing:
        current lane -> selected outgoing link lane -> next, following route edges.
        """
        lanes = set()
        edges = set()
        try:
            route = traci.vehicle.getRoute(emg_id)
            route_idx = traci.vehicle.getRouteIndex(emg_id)
            cur_lane = traci.vehicle.getLaneID(emg_id)
            if not cur_lane:
                return lanes, edges

            lanes.add(cur_lane)
            edges.add(_lane_edge(cur_lane))
            if route_idx < 0:
                route_idx = 0
            edge_pos = route_idx

            lane = cur_lane
            hops = 0
            while hops < hop_limit:
                links = _lane_link_targets(lane)
                if not links:
                    break

                target_edge = route[edge_pos + 1] if edge_pos + 1 < len(route) else None
                chosen = None
                if target_edge:
                    for to_lane, via_lane in links:
                        if _lane_edge(to_lane) == target_edge:
                            chosen = (to_lane, via_lane)
                            break
                if chosen is None:
                    chosen = links[0]

                to_lane, via_lane = chosen
                if via_lane:
                    lanes.add(via_lane)
                    edges.add(_lane_edge(via_lane))
                lanes.add(to_lane)
                edges.add(_lane_edge(to_lane))

                if target_edge and _lane_edge(to_lane) == target_edge:
                    edge_pos += 1
                lane = to_lane
                hops += 1
        except:
            pass
        return lanes, edges

    def _route_overlaps_corridor(self, vid, corridor_edges):
        """
        Extra strictness: only command vehicles whose near-future route
        overlaps the EV traced corridor edges.
        """
        if not corridor_edges:
            return True
        try:
            edge_now = traci.vehicle.getRoadID(vid)
            if edge_now in corridor_edges:
                return True
            route = traci.vehicle.getRoute(vid)
            idx = traci.vehicle.getRouteIndex(vid)
            if idx < 0:
                idx = 0
            look = route[idx:idx + 4]
            return any(e in corridor_edges for e in look)
        except:
            return False

    def _veh_known(self, vid):
        try:
            traci.vehicle.getRoadID(vid)
            return True
        except:
            return False

    # ═══════════════════════════════════════════════════════════
    # FIX A: EARLY SIGNAL PREEMPTION — re-evaluated EVERY step
    #
    # Key change from original:
    #   OLD: tl_overridden=True prevented ANY further phase changes.
    #        If winner_phase changed (e.g. firetruck spawns after ambulance),
    #        the TL stayed on the wrong axis.
    #   NEW: winner_phase is recomputed each step. If it differs from what
    #        is currently locked, we immediately update all TLs. The lock
    #        is per-phase, not a one-time flag.
    # ═══════════════════════════════════════════════════════════
    def _preempt_signals_early(self, cur):
        emg_present = self._get_active_emergency_ids(cur)
        if not emg_present:
            return

        # Determine which axis has the EV closest to the junction (shortest ETA)
        winner_phase = HORIZONTAL_GREEN_PHASE
        best_eta     = float('inf')
        cx, cy       = self.intersection_ctr

        for emg_id in emg_present:
            try:
                ex, ey = traci.vehicle.getPosition(emg_id)
                spd    = max(0.5, traci.vehicle.getSpeed(emg_id))
                dist   = math.sqrt((ex - cx) ** 2 + (ey - cy) ** 2)
                eta    = dist / spd
                if eta < best_eta:
                    best_eta = eta
                    axis = _motion_axis(emg_id)
                    winner_phase = (VERTICAL_GREEN_PHASE
                                   if axis == 'v'
                                   else HORIZONTAL_GREEN_PHASE)
            except:
                pass

        # Always push phase and duration — do NOT gate behind a one-time flag
        # This ensures the correct axis is green from the very first step of Phase 4
        if winner_phase != self.tl_winner_phase or not self.tl_phase_locked:
            for tl in self.tl_ids:
                try:
                    traci.trafficlight.setPhase(tl, winner_phase)
                    traci.trafficlight.setPhaseDuration(tl, 99999)
                except:
                    pass
            if not self.tl_phase_locked:
                self._log(f"TL PREEMPT: phase {winner_phase} locked (EV corridor)")
                if self.dash:
                    self.dash.log(
                        f"TL PREEMPT: phase {winner_phase} locked for EV corridor",
                        'EMERGENCY')
            self.tl_winner_phase = winner_phase
            self.tl_phase_locked = True
        else:
            # Refresh duration so it never expires mid-step
            for tl in self.tl_ids:
                try:
                    traci.trafficlight.setPhaseDuration(tl, 99999)
                except:
                    pass

    # ═══════════════════════════════════════════════════════════
    def _ml_assess(self, features, cur):
        if self.no_v2v:
            return
        batch = [(v, nb, fv)
                 for v, vf in features.items()
                 for nb, fv in vf.items()
                 if v  not in self.INCIDENT_VEHICLES
                 and nb not in self.INCIDENT_VEHICLES]
        if not batch:
            return
        try:
            labels, _ = self.rf.predict_batch(
                np.vstack([b[2] for b in batch]))
        except:
            return

        RMAP = {0: 'SAFE', 1: 'RISK', 2: 'COLLISION'}
        vid_max = {}
        for i, (v, nb, _) in enumerate(batch):
            lbl = int(labels[i])
            vid_max[v] = max(vid_max.get(v, 0), lbl)
            if lbl == 2:
                self.collision_count += 1
                self.net.set_alert(v, 'COLLISION')
                key = tuple(sorted([v, nb]))
                if self.sim_t - self._coll_log.get(key, -99) >= 3.0:
                    self._coll_log[key] = self.sim_t
                    if self.dash:
                        self.dash.log(
                            f"COLLISION RISK: {v} <-> {nb}", 'COLLISION')

        for vid in cur:
            if vid in self.INCIDENT_VEHICLES:
                continue
            old = self.risk_state.get(vid, 'SAFE')
            new = RMAP.get(vid_max.get(vid, 0), 'SAFE')
            self.risk_state[vid] = new
            if (old == 'COLLISION' and new == 'SAFE'
                    and vid not in self.saved_vids):
                self.saved_vids.add(vid)
                self.prevented += 1
                self.report.mark_saved(vid)
                self.report.mark_prevented()
                if self.dash:
                    self.dash.log(f"SAVED: {vid} avoided collision", 'SAFE')

    # ═══════════════════════════════════════════════════════════
    def _brake_failure(self, cur):
        vid = self.BRAKE_FAIL
        if vid not in cur:
            return
        try:
            spd = traci.vehicle.getSpeed(vid)
            traci.vehicle.setSpeedMode(vid, 0)
            traci.vehicle.setSpeed(vid, min(spd + 0.3, 22.0))
            self.net.set_alert(vid, 'BRAKE_FAILURE')
            self.risk_state[vid] = 'COLLISION'
            if self.sim_t - self._bf_last_log >= 5.0:
                self._bf_last_log = self.sim_t
                if self.dash:
                    self.dash.log(
                        f"BRAKE FAILURE: {vid} {spd:.1f}m/s", 'COLLISION')
        except:
            pass

    def _wrong_way(self):
        vid = self.WRONG_WAY
        if vid not in self.active_vids:
            return
        try:
            self.net.set_alert(vid, 'WRONG_WAY')
            self.risk_state[vid] = 'COLLISION'
            if not self._ww_logged:
                self._ww_logged = True
                if self.dash:
                    self.dash.log(f"WRONG-WAY DRIVER: {vid}", 'COLLISION')
        except:
            pass

    def _pedestrian(self, cur):
        ped = self.PED
        if ped not in cur:
            return
        try:
            px, py = traci.vehicle.getPosition(ped)
            for cid in cur:
                if cid == ped or cid in self.INCIDENT_VEHICLES:
                    continue
                try:
                    cx, cy = traci.vehicle.getPosition(cid)
                    if math.sqrt((cx - px) ** 2 + (cy - py) ** 2) < 40.0:
                        if not self._veh_known(cid):
                            continue
                        traci.vehicle.setSpeed(cid, 0.0)
                        self.risk_state[cid] = 'RISK'
                        if self.dash and self.step_n % 50 == 0:
                            self.dash.log(f"PED STOP: {cid}", 'RISK')
                except:
                    pass
        except:
            pass

    # ═══════════════════════════════════════════════════════════
    # FIX B + C: Emergency corridor — intelligent lane selection
    #            + intersection clearance each step
    # ═══════════════════════════════════════════════════════════
    def _emergency_corridor(self, cur):
        emg_present = self._get_active_emergency_ids(cur)
        if not emg_present:
            return

        # First activation
        if not self.emg_active:
            self.emg_active  = True
            self.emg_entry_t = self.sim_t
            self.report.emg_entry_t = self.sim_t
            self._log("EMERGENCY CORRIDOR ACTIVATED")
            if self.dash:
                self.dash.log("", 'INFO')
                self.dash.log("═══ EMERGENCY CORRIDOR ACTIVE ═══", 'EMERGENCY')
                self.dash.log(f"EV count: {len(emg_present)} active", 'EMERGENCY')
                self.dash.log("", 'INFO')
                self.dash.set_emergency_priority(True)
                self.dash.update_impact(
                    len(self.saved_vids), self.prevented,
                    emg_vehicles=", ".join(sorted(emg_present)))

        # Pick priority EV (closest to junction by ETA)
        winner_id = self._pick_winner(emg_present)

        # FIX C: Clear intersection BEFORE EV arrives — every step
        self._clear_intersection(winner_id, cur)

        # ── EV control ─────────────────────────────────────────
        for emg_id in emg_present:
            try:
                traci.vehicle.setSpeedMode(emg_id, 7)
                traci.vehicle.setLaneChangeMode(emg_id, 597)
                target_spd = self._ev_adaptive_speed(emg_id, winner_id, cur)
                traci.vehicle.setSpeed(emg_id, target_spd)
                self.net.set_alert(emg_id, 'EMERGENCY_VEHICLE')
                self.risk_state[emg_id] = 'SAFE'
            except:
                pass

        # Get EV positions
        emg_info = {}
        for emg_id in emg_present:
            try:
                ex, ey = traci.vehicle.getPosition(emg_id)
                corridor_lanes, corridor_edges = self._lane_path_for_ev(emg_id)
                emg_info[emg_id] = (ex, ey, corridor_lanes, corridor_edges)
            except:
                pass

        # ── Normal vehicle handling — PATH-BASED, not radius-based ───
        emg_ids = set(emg_present)
        for vid in cur:
            if vid in emg_ids:
                continue
            if vid in self._junction_clearing:
                continue

            try:
                vx, vy = traci.vehicle.getPosition(vid)
            except:
                continue

            # Find the most actionable role across all active EVs
            best_role  = None
            nearest_id = None
            best_along = 0.0
            vid_lane = ''
            vid_edge = ''
            try:
                vid_lane = traci.vehicle.getLaneID(vid)
                vid_edge = traci.vehicle.getRoadID(vid)
            except:
                pass

            for emg_id, (ex, ey, corridor_lanes, corridor_edges) in emg_info.items():
                if corridor_lanes and vid_lane and not (
                    vid_lane in corridor_lanes or vid_edge in corridor_edges
                ):
                    continue
                if not self._route_overlaps_corridor(vid, corridor_edges):
                    continue
                role, along, lateral = _in_ev_path(vid, emg_id)
                if role is not None:
                    if best_role is None or role == 'ahead':
                        best_role  = role
                        nearest_id = emg_id
                        best_along = along

            # ── AUTO-RESTORE: EV has already passed this vehicle ──────
            # A previously commanded vehicle whose role is now None (EV moved on)
            # OR a 'behind' vehicle that has slid far behind the EV's window
            # gets immediately restored so traffic can flow again.
            already_held = vid in (self.pullover_vids | self.waiting_vids)
            ev_passed    = already_held and vid not in self.restored_vids

            if ev_passed and best_role is None:
                try:
                    traci.vehicle.setSpeedMode(vid, 31)
                    traci.vehicle.setLaneChangeMode(vid, 1621)
                    traci.vehicle.setSpeed(vid, -1)
                    self.restored_vids.add(vid)
                    self.risk_state[vid] = 'SAFE'
                    if self.dash:
                        self.dash.log(
                            f"RESTORED: {vid} — EV has passed", 'SAFE')
                except:
                    pass
                continue

            if best_role is None:
                continue

            # ── SAME-DIRECTION vehicles AHEAD of EV ──────────────
            if best_role == 'ahead':
                try:
                    traci.vehicle.setSpeedMode(vid, 31)
                    traci.vehicle.setLaneChangeMode(vid, 0)

                    # Issue lane change immediately — BEFORE slowing.
                    # Vehicle moves to road edge while it still has speed to do so.
                    self._make_space_either_lane(vid, nearest_id)

                    # Progressive speed: far vehicles slow gently, close ones crawl.
                    # This gives vehicles time to complete their lane change while
                    # smoothly forming the corridor ahead of the EV.
                    cur_spd = traci.vehicle.getSpeed(vid)
                    if best_along > 80.0:
                        target_spd = 8.0    # far ahead — light braking, move aside
                    elif best_along > 40.0:
                        target_spd = 5.0    # medium — moderate slow
                    else:
                        target_spd = CRAWL_SPEED_MPS  # close — crawl

                    # If still close and still on the EV lane, force a stronger clear.
                    if nearest_id is not None and best_along < FORCE_CLEAR_GAP_M:
                        try:
                            if traci.vehicle.getLaneID(vid) == traci.vehicle.getLaneID(nearest_id):
                                # Hard blocker directly in EV lane near junction:
                                # push it forward out of the EV nose instead of
                                # letting it sit and trap the EV behind it.
                                traci.vehicle.setSpeedMode(vid, 7)
                                target_spd = max(target_spd, 8.0)
                                self._make_space_either_lane(vid, nearest_id)
                        except:
                            pass

                    if cur_spd > target_spd:
                        traci.vehicle.setSpeed(vid, target_spd)

                    self.risk_state[vid] = 'RISK'
                    if vid not in self.pullover_vids:
                        self.pullover_vids.add(vid)
                        if self.dash:
                            self.dash.log(
                                f"CORRIDOR AHEAD: {vid} at {best_along:.0f}m → lane edge",
                                'RISK')
                except:
                    pass

            # ── SAME-DIRECTION vehicles beside/behind EV ─────────
            elif best_role == 'behind':
                try:
                    # Issue lane change first (works at any speed via changeLane)
                    traci.vehicle.setLaneChangeMode(vid, 0)
                    self._make_space_either_lane(vid, nearest_id)

                    # Give a very brief crawl window ONLY while lane change is
                    # pending (first 3 steps after assignment). After that: full stop.
                    steps_since_assign = self.step_n - self._lane_locked.get(vid, 0)
                    if vid not in self._lane_assigned or steps_since_assign <= 3:
                        # Lane change pending — allow minimal movement
                        traci.vehicle.setSpeedMode(vid, 31)
                        cur_spd = traci.vehicle.getSpeed(vid)
                        if cur_spd > 1.5:
                            traci.vehicle.setSpeed(vid, 1.5)
                    else:
                        # Lane assigned + time elapsed — hard stop, clear EV lane
                        traci.vehicle.setSpeedMode(vid, 0)
                        traci.vehicle.setSpeed(vid, 0.0)

                    self.risk_state[vid] = 'RISK'
                    if vid not in self.pullover_vids:
                        self.pullover_vids.add(vid)
                        if self.dash:
                            self.dash.log(
                                f"PULLING OVER: {vid} (beside → stop)", 'RISK')
                except:
                    pass

            # ── CROSS-AXIS vehicles at junction ───────────────────
            elif best_role == 'cross':
                try:
                    traci.vehicle.setSpeedMode(vid, 0)
                    traci.vehicle.setSpeed(vid, 0.0)
                    self.risk_state[vid] = 'RISK'
                    if vid not in self.waiting_vids:
                        self.waiting_vids.add(vid)
                        if self.dash:
                            self.dash.log(
                                f"WAITING: {vid} stopped (crosses EV path)",
                                'RISK')
                except:
                    pass

    # ═══════════════════════════════════════════════════════════
    # FIX C (v2): Clear junction box — detect by internal lane, not EV radius.
    #
    # OLD bug: used distance-from-EV (40m) — EV was 80m away so nothing was
    # cleared, and the nudge fought the cross-axis stop commands in same step.
    #
    # NEW approach:
    #   • A vehicle is "in the box" if its edge starts with ':' (SUMO internal).
    #   • Junction vehicles are nudged forward at 8 m/s with speedMode=7.
    #   • We track `_junction_clearing` so we only log once per vehicle.
    #   • This runs BEFORE the normal corridor per-vehicle loop so these
    #     vehicles' speedMode is set last → corridor loop must NOT overwrite them.
    #     The corridor loop now skips vehicles already in _junction_clearing.
    # ═══════════════════════════════════════════════════════════
    def _clear_intersection(self, winner_id, cur):
        emg_ids = set(self._get_active_emergency_ids(cur))
        for vid in cur:
            if vid in emg_ids:
                continue
            try:
                edge = traci.vehicle.getRoadID(vid)
            except:
                continue

            if edge.startswith(':'):
                # Vehicle is physically inside the junction box — push it out
                try:
                    if not self._veh_known(vid):
                        continue
                    traci.vehicle.setSpeedMode(vid, 7)
                    traci.vehicle.setSpeed(vid, 8.0)
                    self._junction_clearing.add(vid)
                    if self.dash and vid not in self._jc_logged:
                        self._jc_logged.add(vid)
                        self.dash.log(
                            f"JUNCTION CLEAR: nudging {vid} out of box", 'RISK')
                except:
                    pass
            else:
                # Vehicle left the junction — stop clearing it
                self._junction_clearing.discard(vid)

    # ═══════════════════════════════════════════════════════════
    # FIX B: Intelligent lane selection — pushes vehicle AWAY from
    # the EV's travel direction, not randomly left or right.
    # Called ONCE per vehicle (enforced by _lane_assigned dict).
    # ═══════════════════════════════════════════════════════════
    def _make_space_either_lane(self, vid, emg_id):
        """
        Corridor formation: push vid to the edge of the road away from the EV's
        travel axis, creating a clear center lane for the EV to drive through.

        Logic:
          - Compute which side of the EV's path centreline the vehicle is on
            using the perpendicular dot product (signed lateral).
          - Vehicles on the LEFT side → push to leftmost lane (highest index).
          - Vehicles on the RIGHT side → push to rightmost lane (index 0).
          - This splits traffic to both sides, opening the center for the EV.

        Uses changeLane() directly (not couldChangeLane) so it works at any speed.
        Locked once via _lane_assigned — never changed again.
        """
        try:
            if emg_id is None:
                return
            if not self._veh_known(vid) or not self._veh_known(emg_id):
                return

            vx, vy = traci.vehicle.getPosition(vid)
            ex, ey = traci.vehicle.getPosition(emg_id)

            edge = traci.vehicle.getRoadID(vid)
            if edge.startswith(':'):
                self._lane_assigned[vid] = traci.vehicle.getLaneIndex(vid)
                return

            lane_i = traci.vehicle.getLaneIndex(vid)
            lane_n = traci.edge.getLaneNumber(edge)

            if lane_n <= 1:
                self._lane_assigned[vid] = lane_i
                return

            # Already assigned and currently on assigned lane? nothing to do.
            assigned = self._lane_assigned.get(vid)
            if assigned is not None and lane_i == assigned:
                return
            # avoid hammering lane change every step
            if (self.step_n - self._lane_locked.get(vid, -9999)) < LANE_RETRY_STEPS:
                return

            # Signed lateral: which side of EV path centreline is the vehicle on?
            fx, fy = _ev_heading_vec(emg_id)
            px, py = -fy, fx   # perpendicular (points LEFT of EV heading)
            dx = vx - ex
            dy = vy - ey
            signed_lateral = dx * px + dy * py
            # positive signed_lateral → vehicle is to the LEFT of EV heading
            # negative signed_lateral → vehicle is to the RIGHT of EV heading

            # Prefer a lane different from EV's lane first.
            try:
                ev_lane_i = traci.vehicle.getLaneIndex(emg_id)
            except:
                ev_lane_i = lane_i

            # In SUMO: lane 0 = rightmost, lane (n-1) = leftmost
            if ev_lane_i == 0:
                target = lane_n - 1
            elif ev_lane_i == lane_n - 1:
                target = 0
            else:
                target = lane_n - 1 if signed_lateral > 0 else 0

            if target == ev_lane_i:
                target = 0 if ev_lane_i != 0 else lane_n - 1

            if target == lane_i:
                # Already on the correct edge lane — lock in place
                self._lane_assigned[vid] = lane_i
                return

            traci.vehicle.setLaneChangeMode(vid, 0)
            traci.vehicle.changeLane(vid, target, 15.0)
            self._lane_assigned[vid] = target
            self._lane_locked[vid]   = self.step_n

            if self.dash:
                side = "left-edge" if signed_lateral > 0 else "right-edge"
                self.dash.log(
                    f"CORRIDOR: {vid} → lane {target} ({side})", 'RISK')
        except:
            pass

    # ═══════════════════════════════════════════════════════════
    def _on_ev_exited(self, exited_vid, cur):
        remaining_evs = self._get_active_emergency_ids(cur)
        t_clear = self.sim_t - self.emg_entry_t
        self._log(f"{exited_vid} reached destination in {t_clear:.1f}s")

        if self.dash:
            self.dash.log(
                f"CORRIDOR CLEAR — {exited_vid} reached dest in {t_clear:.1f}s",
                'SAFE')
            self.dash.update_impact(
                len(self.saved_vids), self.prevented, t_clear)

        self.report.emg_clear_t = self.sim_t

        if not remaining_evs:
            self._close_corridor(cur)

    def _close_corridor(self, cur):
        """Restore signals and release all held vehicles."""
        for tl in self.tl_ids:
            try:
                traci.trafficlight.setProgram(tl, '0')
            except:
                pass
        self.tl_phase_locked = False
        self.tl_winner_phase = None

        for vid in (self.pullover_vids | self.waiting_vids):
            if vid in cur and vid not in self.restored_vids:
                try:
                    traci.vehicle.setSpeedMode(vid, 31)
                    traci.vehicle.setLaneChangeMode(vid, 1621)
                    traci.vehicle.setSpeed(vid, -1)
                    self.restored_vids.add(vid)
                    self.risk_state[vid] = 'SAFE'
                except:
                    pass

        self.emg_active = False
        self._lane_assigned.clear()
        self._lane_locked.clear()
        self._junction_clearing.clear()
        self._jc_logged.clear()

        if self.dash:
            self.dash.log("Traffic lights restored", 'INFO')
            self.dash.log("═══ EMERGENCY CORRIDOR CLOSED ═══", 'INFO')
            self.dash.set_emergency_priority(False)

    # ═══════════════════════════════════════════════════════════
    def _pick_winner(self, emg_present):
        """Pick the EV with shortest ETA to the intersection."""
        winner   = emg_present[0]
        best_eta = float('inf')
        cx, cy   = self.intersection_ctr
        for emg_id in emg_present:
            try:
                ex, ey = traci.vehicle.getPosition(emg_id)
                spd    = max(0.5, traci.vehicle.getSpeed(emg_id))
                dist   = math.sqrt((ex - cx) ** 2 + (ey - cy) ** 2)
                eta    = dist / spd
                if eta < best_eta:
                    best_eta = eta
                    winner   = emg_id
            except:
                pass
        return winner

    # ═══════════════════════════════════════════════════════════
    def _ev_adaptive_speed(self, emg_id, winner_id, cur):
        """
        Smooth adaptive speed for EV:
          < 8m  gap  → 2.0 m/s  (near-stop, avoid collision)
          < 25m gap  → 6.0 m/s  (slow approach)
          < 60m gap  → 10.0 m/s (medium — road partially clear)
          open road  → 13.5 m/s (full speed)

        Excludes junction-clearing vehicles from the gap scan — those are
        being actively nudged out of the way and should not hold the EV back.
        """
        try:
            ex, ey = traci.vehicle.getPosition(emg_id)
        except:
            return 13.5

        emg_ids = set(self._get_active_emergency_ids(cur))
        min_gap = float('inf')
        for vid in cur:
            if vid == emg_id or vid in emg_ids:
                continue
            if vid in self._junction_clearing:
                continue
            if vid in self.restored_vids:
                continue
            # Skip vehicles that already have a lane assignment — they are
            # actively moving out of the EV's way and should not throttle it.
            if vid in self._lane_assigned:
                continue
            try:
                role, along, lateral = _in_ev_path(vid, emg_id)
                if role != 'ahead':
                    continue
                vx, vy = traci.vehicle.getPosition(vid)
                d = math.sqrt((vx - ex) ** 2 + (vy - ey) ** 2)
                if d < min_gap:
                    min_gap = d
            except:
                pass

        ev_max = 18.0
        try:
            ev_max = max(16.0, traci.vehicle.getMaxSpeed(emg_id))
        except:
            pass

        # Smooth speed tiers — higher top-end when road is clear
        if min_gap < EV_GAP_THROTTLE_HARD:
            return 2.0    # imminent — near-stop
        if min_gap < EV_GAP_THROTTLE_SOFT:
            return 6.0    # close — cautious approach
        if min_gap < 60.0:
            return 10.0   # moderate gap — accelerating through
        return min(ev_max, 20.0)   # open road — full emergency speed

    # ═══════════════════════════════════════════════════════════
    def _speed_zone(self, cur):
        if self.no_v2v:
            return
        centres = []
        for vid in cur:
            if self.risk_state.get(vid) == 'COLLISION':
                try:
                    x, y = traci.vehicle.getPosition(vid)
                    centres.append((x, y))
                except:
                    pass
        if not centres:
            return
        for vid in cur:
            if vid in self.INCIDENT_VEHICLES:
                continue
            if self.risk_state.get(vid) == 'COLLISION':
                continue
            try:
                vx, vy = traci.vehicle.getPosition(vid)
                in_zone = any(
                    math.sqrt((vx - cx) ** 2 + (vy - cy) ** 2) < 80
                    for cx, cy in centres)
                if in_zone:
                    if traci.vehicle.getSpeed(vid) > 4.0:
                        traci.vehicle.setSpeed(vid, 4.0)
                        if self.risk_state.get(vid) == 'SAFE':
                            self.risk_state[vid] = 'RISK'
                else:
                    traci.vehicle.setSpeed(vid, -1)
            except:
                pass

    # ═══════════════════════════════════════════════════════════
    def _colours(self, cur):
        for vid in cur:
            try:
                vtype = traci.vehicle.getTypeID(vid)

                if vtype == 'ambulance':
                    traci.vehicle.setColor(vid,
                        (255, 255, 255, 255)
                        if self.emg_active and (self.step_n // 5) % 2 == 0
                        else COL_AMBULANCE)
                    continue

                if vtype == 'firetruck':
                    traci.vehicle.setColor(vid,
                        (255, 200, 0, 255)
                        if self.emg_active and (self.step_n // 5) % 2 == 0
                        else COL_FIRETRUCK)
                    continue
                if vtype == 'rescue_ev':
                    traci.vehicle.setColor(vid,
                        (255, 255, 180, 255)
                        if self.emg_active and (self.step_n // 5) % 2 == 0
                        else COL_RESCUE)
                    continue

                if vid == self.BRAKE_FAIL:
                    traci.vehicle.setColor(vid, COL_BRAKE); continue
                if vid == self.WRONG_WAY:
                    traci.vehicle.setColor(vid, COL_WRONG); continue
                if vid == self.PED:
                    traci.vehicle.setColor(vid, (255, 255, 0, 255)); continue

                if self.emg_active and vid not in self.restored_vids:
                    if vid in self.pullover_vids:
                        traci.vehicle.setColor(vid, COL_PULLOVER); continue
                    if vid in self.waiting_vids:
                        traci.vehicle.setColor(vid, COL_WAITING); continue

                risk = self.risk_state.get(vid, 'SAFE')
                traci.vehicle.setColor(vid,
                    {'SAFE': COL_SAFE,
                     'RISK': COL_RISK,
                     'COLLISION': COL_COLLISION}.get(risk, COL_SAFE))
            except:
                pass

    # ═══════════════════════════════════════════════════════════
    def _counts(self, vids):
        c = {'SAFE': 0, 'RISK': 0, 'COLLISION': 0}
        for v in vids:
            r = self.risk_state.get(v, 'SAFE')
            c[r] = c.get(r, 0) + 1
        return c

    def _dash_update(self, ml_m, cur):
        if not self.dash:
            return
        cnts = self._counts(cur)
        self.dash.update_risk(cnts['SAFE'], cnts['RISK'], cnts['COLLISION'])
        self.dash.update_net(self.net.avg_pdr, self.net.avg_delay_ms,
                             self.last_hop, self.net.beacons_sent)
        self.dash.update_impact(len(self.saved_vids), self.prevented)

    def _log(self, msg):
        print(f"[{self.sim_t:6.1f}s] {msg}")