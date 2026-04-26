"""
V2V Network Layer — IEEE 802.11p beacon simulation,
multi-hop alert relay, RSU coordination.
"""
import numpy as np
import time
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional

COMM_RANGE   = 150.0   # metres  (IEEE 802.11p typical)
RSU_RANGE    = 250.0   # metres  (RSU has wider coverage)
BEACON_HZ    = 10      # beacons per second
HOP_LIMIT    = 5

# RSU positions (corners of intersection)
RSU_POSITIONS = {
    'RSU_NE': ( 60,  60),
    'RSU_NW': (-60,  60),
    'RSU_SE': ( 60, -60),
    'RSU_SW': (-60, -60),
}

@dataclass
class Beacon:
    vid:        str
    t:          float
    x:          float
    y:          float
    speed:      float
    accel:      float
    direction:  float
    risk:       str   = 'SAFE'   # SAFE / RISK / COLLISION
    alert:      bool  = False
    alert_type: str   = ''
    hop:        int   = 0
    seq:        int   = 0
    ttl:        int   = HOP_LIMIT
    is_emergency: bool = False
    emergency_role: str = ''   # e.g., AMBULANCE / FIRETRUCK

@dataclass
class VehicleState:
    vid:        str
    x:          float = 0.0
    y:          float = 0.0
    speed:      float = 0.0
    accel:      float = 0.0
    direction:  float = 0.0
    risk:       str   = 'SAFE'
    history:    list  = field(default_factory=list)
    seq:        int   = 0
    alert_active: bool = False
    alert_type:   str  = ''
    seen_alerts:  Set  = field(default_factory=set)
    neighbors:    Dict = field(default_factory=dict)  # vid -> Beacon
    is_emergency: bool = False
    emergency_role: str = ''

class V2VNetwork:
    def __init__(self, plr: float = 0.0, seed: int = 42):
        self.plr   = plr
        self.rng   = np.random.RandomState(seed)
        self.vehicles: Dict[str, VehicleState] = {}
        self.rsus  = RSU_POSITIONS
        # Stats
        self.beacons_sent     = 0
        self.beacons_received = 0
        self.beacons_lost     = 0
        self.delays_ms: List[float] = []
        self.alert_log: List[Dict]  = []
        self.step_pdrs: List[float] = []

    # ── vehicle registration ──────────────────────────────────
    def register(self, vid: str):
        if vid not in self.vehicles:
            self.vehicles[vid] = VehicleState(vid=vid)

    def remove(self, vid: str):
        self.vehicles.pop(vid, None)

    def update_state(self, vid: str, x, y, speed, accel, direction,
                     is_emergency: bool = False, emergency_role: str = ''):
        if vid not in self.vehicles: self.register(vid)
        vs = self.vehicles[vid]
        vs.x=x; vs.y=y; vs.speed=speed; vs.accel=accel; vs.direction=direction
        vs.is_emergency = is_emergency
        vs.emergency_role = emergency_role
        vs.history.append({'x':x,'y':y,'speed':speed,'direction':direction,'t':time.time()})
        if len(vs.history) > 100: vs.history.pop(0)

    # ── distance helpers ─────────────────────────────────────
    def _dist(self, x1,y1, x2,y2): return np.sqrt((x1-x2)**2+(y1-y2)**2)
    def _ttc(self, vs: VehicleState, nb: Beacon) -> float:
        dist = self._dist(vs.x,vs.y,nb.x,nb.y)
        vx1  = vs.speed*np.cos(np.radians(vs.direction))
        vy1  = vs.speed*np.sin(np.radians(vs.direction))
        vx2  = nb.speed*np.cos(np.radians(nb.direction))
        vy2  = nb.speed*np.sin(np.radians(nb.direction))
        rel  = np.sqrt((vx1-vx2)**2+(vy1-vy2)**2)
        return dist/rel if rel > 0.1 else 999.0

    # ── channel model ─────────────────────────────────────────
    def _deliver(self, dist: float) -> Tuple[bool, float]:
        if dist > COMM_RANGE: return False, 0.0
        dist_plr = 0.02*(dist/COMM_RANGE)**2
        eff_plr  = min(1.0, self.plr + dist_plr + self.rng.exponential(0.005))
        if self.rng.random() < eff_plr: return False, 0.0
        delay_ms = (dist/3e8)*1000 + self.rng.uniform(0.4,1.8)
        return True, delay_ms

    # ── RSU boost ────────────────────────────────────────────
    def _rsu_in_range(self, x, y) -> List[str]:
        in_range = []
        for rsu_id,(rx,ry) in self.rsus.items():
            if self._dist(x,y,rx,ry) <= RSU_RANGE:
                in_range.append(rsu_id)
        return in_range

    # ── main step ────────────────────────────────────────────
    def step(self) -> Dict[str, Dict]:
        """
        Broadcast beacons from all vehicles, deliver to neighbours,
        run RSU relay, return feature vectors for ML prediction.
        """
        vlist = list(self.vehicles.values())
        beacons: Dict[str,Beacon] = {}

        # Create beacons
        for vs in vlist:
            vs.seq += 1
            b = Beacon(vid=vs.vid, t=time.time(),
                       x=vs.x, y=vs.y, speed=vs.speed,
                       accel=vs.accel, direction=vs.direction,
                       risk=vs.risk, alert=vs.alert_active,
                       alert_type=vs.alert_type,
                       hop=0, seq=vs.seq, ttl=HOP_LIMIT,
                       is_emergency=vs.is_emergency,
                       emergency_role=vs.emergency_role)
            beacons[vs.vid] = b
            self.beacons_sent += 1

        # Direct V2V delivery
        received = 0; lost = 0
        for vs in vlist:
            vs.neighbors.clear()
            for other_id, b in beacons.items():
                if other_id == vs.vid: continue
                dist = self._dist(vs.x,vs.y,b.x,b.y)
                ok, delay = self._deliver(dist)
                if ok:
                    vs.neighbors[other_id] = b
                    self.delays_ms.append(delay)
                    received += 1
                else:
                    lost += 1

        # RSU relay — vehicles near RSU get extra reach
        for vs in vlist:
            rsu_ids = self._rsu_in_range(vs.x, vs.y)
            if not rsu_ids: continue
            for other in vlist:
                if other.vid in vs.neighbors or other.vid == vs.vid: continue
                # Check if other vehicle is also near any RSU
                other_rsus = self._rsu_in_range(other.x, other.y)
                if set(rsu_ids) & set(other_rsus):  # share an RSU
                    b = beacons.get(other.vid)
                    if b:
                        vs.neighbors[other.vid] = b
                        received += 1

        # Multi-hop alert relay
        alert_beacons = [b for b in beacons.values() if b.alert and b.ttl > 0]
        # Dedicated emergency status beacon in addition to standard state beacons.
        # This gives nearby vehicles explicit emergency intent + kinematics.
        for b in beacons.values():
            if b.is_emergency:
                emg_b = Beacon(**b.__dict__)
                emg_b.alert = True
                emg_b.alert_type = 'EMERGENCY_STATUS'
                emg_b.ttl = HOP_LIMIT
                alert_beacons.append(emg_b)
        for ab in alert_beacons:
            self._multihop_relay(ab, vlist, set())

        self.beacons_received += received
        self.beacons_lost     += lost
        pdr = received / max(received+lost, 1)
        self.step_pdrs.append(pdr)

        # Build feature vectors for ML
        features = {}
        for vs in vlist:
            vf = {}
            for nb_id, nb in vs.neighbors.items():
                dist     = self._dist(vs.x,vs.y,nb.x,nb.y)
                vx1      = vs.speed*np.cos(np.radians(vs.direction))
                vy1      = vs.speed*np.sin(np.radians(vs.direction))
                vx2      = nb.speed*np.cos(np.radians(nb.direction))
                vy2      = nb.speed*np.sin(np.radians(nb.direction))
                rel_spd  = np.sqrt((vx1-vx2)**2+(vy1-vy2)**2)
                ttc      = self._ttc(vs, nb)
                dd       = abs(vs.direction - nb.direction)%360
                if dd > 180: dd = 360-dd
                vf[nb_id] = np.array([
                    dist, rel_spd, vs.speed, nb.speed,
                    vs.accel, nb.accel, min(ttc,100), dd
                ])
            features[vs.vid] = vf
        return features

    def _multihop_relay(self, alert: Beacon, vlist, seen: Set):
        # Include alert type so emergency-status relay does not get deduplicated
        # against a regular alert beacon from the same sender/sequence.
        key = f"{alert.vid}_{alert.seq}_{alert.alert_type}"
        if key in seen or alert.ttl <= 0: return
        seen.add(key)
        relayed = Beacon(**alert.__dict__)
        relayed.hop += 1; relayed.ttl -= 1
        for vs in vlist:
            if vs.vid == alert.vid: continue
            dist = self._dist(vs.x,vs.y,alert.x,alert.y)
            ok,_ = self._deliver(dist)
            if ok:
                if key not in vs.seen_alerts:
                    vs.seen_alerts.add(key)
                    # log it
                    self.alert_log.append({
                        't': time.time(),
                        'from': alert.vid,
                        'to':   vs.vid,
                        'type': alert.alert_type,
                        'hop':  relayed.hop,
                        'is_emergency': alert.is_emergency,
                        'emergency_role': alert.emergency_role
                    })
                    self._multihop_relay(relayed, vlist, seen)

    def set_alert(self, vid: str, alert_type: str):
        if vid in self.vehicles:
            self.vehicles[vid].alert_active = True
            self.vehicles[vid].alert_type   = alert_type

    def clear_alert(self, vid: str):
        if vid in self.vehicles:
            self.vehicles[vid].alert_active = False
            self.vehicles[vid].alert_type   = ''

    @property
    def avg_pdr(self): return float(np.mean(self.step_pdrs)) if self.step_pdrs else 0.0
    @property
    def avg_delay_ms(self): return float(np.mean(self.delays_ms)) if self.delays_ms else 0.0