"""
ML Models: Random Forest risk classifier + LSTM trajectory predictor.
Trained offline, used live during simulation via TraCI.
"""
import numpy as np
import pickle, os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import warnings; warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
#  TRAINING DATA GENERATOR
# ─────────────────────────────────────────────────────────────
def generate_training_data(n=8000, seed=42):
    rng = np.random.RandomState(seed)
    X, y = [], []
    for _ in range(n):
        sc = rng.choice(['safe','risk','collision'], p=[0.55,0.30,0.15])
        if sc == 'safe':
            dist      = rng.uniform(40, 300)
            rel_spd   = rng.uniform(0, 5)
            spd_ego   = rng.uniform(3, 14)
            spd_nb    = rng.uniform(3, 14)
            acc_ego   = rng.uniform(-1, 1)
            acc_nb    = rng.uniform(-1, 1)
            ttc       = rng.uniform(10, 100)
            dir_diff  = rng.uniform(0, 40)
            label = 0
        elif sc == 'risk':
            dist      = rng.uniform(10, 60)
            rel_spd   = rng.uniform(4, 16)
            spd_ego   = rng.uniform(8, 22)
            spd_nb    = rng.uniform(8, 22)
            acc_ego   = rng.uniform(-4, 0.5)
            acc_nb    = rng.uniform(-2, 2)
            ttc       = rng.uniform(2, 10)
            dir_diff  = rng.uniform(20, 130)
            label = 1
        else:
            dist      = rng.uniform(0.5, 12)
            rel_spd   = rng.uniform(8, 28)
            spd_ego   = rng.uniform(12, 22)
            spd_nb    = rng.uniform(12, 22)
            acc_ego   = rng.uniform(-9, 0)
            acc_nb    = rng.uniform(-2, 4)
            ttc       = rng.uniform(0, 2.5)
            dir_diff  = rng.uniform(60, 180)
            label = 2
        row = [dist, rel_spd, spd_ego, spd_nb, acc_ego, acc_nb, ttc, dir_diff]
        row = [v + rng.normal(0, 0.015*abs(v)+0.05) for v in row]
        X.append(row); y.append(label)
    return np.array(X), np.array(y)

# ─────────────────────────────────────────────────────────────
#  RANDOM FOREST RISK CLASSIFIER
# ─────────────────────────────────────────────────────────────
class RiskClassifier:
    LABELS = ['SAFE', 'RISK', 'COLLISION']
    FEATURES = ['distance_m','rel_speed_mps','speed_ego','speed_neighbor',
                'accel_ego','accel_neighbor','ttc_s','dir_diff_deg']

    def __init__(self):
        self.model   = RandomForestClassifier(
            n_estimators=200, max_depth=14, min_samples_leaf=2,
            class_weight='balanced', random_state=42, n_jobs=1)
        self.scaler  = StandardScaler()
        self.trained = False
        self.metrics = {}

    def train(self, n_samples=8000):
        print("[RF] Generating training data...")
        X, y = generate_training_data(n_samples)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                               stratify=y, random_state=42)
        Xtr_s = self.scaler.fit_transform(Xtr)
        Xte_s = self.scaler.transform(Xte)
        print("[RF] Training Random Forest...")
        self.model.fit(Xtr_s, ytr)
        self.trained = True
        yp   = self.model.predict(Xte_s)
        yprob= self.model.predict_proba(Xte_s)
        auc  = roc_auc_score(yte, yprob, multi_class='ovr', average='macro')
        acc  = float((yp == yte).mean())
        self.metrics = {'auc': auc, 'accuracy': acc,
                        'report': classification_report(yte, yp,
                                  target_names=self.LABELS)}
        print(f"[RF] AUC={auc:.4f}  Acc={acc:.4f}")
        return self.metrics

    def predict_batch(self, feature_matrix: np.ndarray):
        """Predict labels + probs for a batch of feature rows."""
        if not self.trained:
            raise RuntimeError("Model not trained")
        fs = self.scaler.transform(feature_matrix)
        labels = self.model.predict(fs)
        probs  = self.model.predict_proba(fs)
        return labels, probs

    def save(self, path='models/rf_model.pkl'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path,'wb') as f:
            pickle.dump({'model':self.model,'scaler':self.scaler,
                         'metrics':self.metrics}, f)
        print(f"[RF] Saved → {path}")

    def load(self, path='models/rf_model.pkl'):
        with open(path,'rb') as f:
            d = pickle.load(f)
        self.model=d['model']; self.scaler=d['scaler']
        self.metrics=d.get('metrics',{}); self.trained=True
        print(f"[RF] Loaded ← {path}")

# ─────────────────────────────────────────────────────────────
#  LSTM TRAJECTORY PREDICTOR  (pure NumPy)
# ─────────────────────────────────────────────────────────────
class LSTMCell:
    def __init__(self, in_sz, hid_sz, seed=0):
        rng = np.random.RandomState(seed)
        s   = np.sqrt(1.0/hid_sz); n = in_sz+hid_sz
        self.Wf=rng.uniform(-s,s,(n,hid_sz)); self.bf=np.ones(hid_sz)*0.1
        self.Wi=rng.uniform(-s,s,(n,hid_sz)); self.bi=np.zeros(hid_sz)
        self.Wc=rng.uniform(-s,s,(n,hid_sz)); self.bc=np.zeros(hid_sz)
        self.Wo=rng.uniform(-s,s,(n,hid_sz)); self.bo=np.zeros(hid_sz)
    def _sig(self,x): return 1/(1+np.exp(-np.clip(x,-20,20)))
    def forward(self,x,h,c):
        xh=np.concatenate([x,h])
        f=self._sig(xh@self.Wf+self.bf); i=self._sig(xh@self.Wi+self.bi)
        ct=np.tanh(xh@self.Wc+self.bc); o=self._sig(xh@self.Wo+self.bo)
        c2=f*c+i*ct; h2=o*np.tanh(c2)
        return h2,c2

class TrajectoryPredictor:
    SEQ=20; HID=48; IN=4; OUT=3   # x,y,speed,dir → predict x,y,speed

    def __init__(self):
        self.lstm  = LSTMCell(self.IN, self.HID)
        self.Wo    = np.random.randn(self.HID, self.OUT)*0.01
        self.bo    = np.zeros(self.OUT)
        self.mu    = None; self.sigma = None
        self.trained = False; self.r2 = 0.0

    def _gen_traj(self, n=60, seed=0):
        rng=np.random.RandomState(seed); trajs=[]
        for _ in range(n):
            x,y,spd,d = rng.uniform(0,300),rng.uniform(0,10),rng.uniform(3,14),rng.uniform(0,360)
            t=[]
            for _ in range(70):
                x+=spd*np.cos(np.radians(d))*0.1; y+=spd*np.sin(np.radians(d))*0.1
                spd=max(0,spd+rng.normal(0,0.2)); d=(d+rng.normal(0,2))%360
                t.append([x,y,spd,d])
            trajs.append(np.array(t))
        return trajs

    def train(self, epochs=8, lr=0.008):
        trajs = self._gen_traj(60)
        all_d = np.vstack(trajs)
        self.mu=all_d.mean(0); self.sigma=all_d.std(0)+1e-8
        Xs,Ys=[],[]
        for t in trajs:
            n=(t-self.mu)/self.sigma
            for i in range(len(n)-self.SEQ):
                Xs.append(n[i:i+self.SEQ]); Ys.append(n[i+self.SEQ,:3])
        Xs=np.array(Xs[:500]); Ys=np.array(Ys[:500])
        print(f"[LSTM] Training on {len(Xs)} seqs, {epochs} epochs...")
        for ep in range(epochs):
            loss=0
            for i in np.random.permutation(len(Xs)):
                h=np.zeros(self.HID); c=np.zeros(self.HID)
                for t in range(self.SEQ): h,c=self.lstm.forward(Xs[i,t],h,c)
                pred=h@self.Wo+self.bo; err=pred-Ys[i]
                loss+=0.5*np.sum(err**2)
                self.Wo-=lr*np.outer(h,err); self.bo-=lr*err
                dh=err@self.Wo.T
                xh=np.concatenate([Xs[i,-1],h])
                self.lstm.Wf-=lr*0.01*np.outer(xh,dh)
        preds=[self._run_seq(Xs[i]) for i in range(len(Xs))]
        ss_res=np.sum((Ys-preds)**2); ss_tot=np.sum((Ys-Ys.mean(0))**2)
        self.r2=max(0.0, 1-ss_res/max(ss_tot,1e-8))
        self.trained=True
        print(f"[LSTM] R²={self.r2:.4f}")

    def _run_seq(self, seq):
        h=np.zeros(self.HID); c=np.zeros(self.HID)
        for t in range(self.SEQ): h,c=self.lstm.forward(seq[t],h,c)
        return h@self.Wo+self.bo

    def predict(self, history, horizon=7):
        """history: list of dicts with x,y,speed,direction. Returns list of predictions."""
        if not self.trained or len(history)<5: return []
        raw=np.array([[d['x'],d['y'],d['speed'],d['direction']] for d in history[-self.SEQ:]])
        if len(raw)<self.SEQ:
            raw=np.vstack([np.tile(raw[0],(self.SEQ-len(raw),1)),raw])
        if self.mu is not None: norm=(raw-self.mu)/self.sigma
        else: norm=raw.copy()
        h=np.zeros(self.HID); c=np.zeros(self.HID)
        for t in range(self.SEQ): h,c=self.lstm.forward(norm[t],h,c)
        preds=[]
        last=raw[-1].copy()
        for step in range(horizon):
            out=h@self.Wo+self.bo
            if self.mu is not None:
                out=out*self.sigma[:3]+self.mu[:3]
            px,py,ps=float(out[0]),float(out[1]),max(0,float(out[2]))
            preds.append({'step':step+1,'x':px,'y':py,'speed':ps,'t_ahead':(step+1)*0.1})
            nd=np.degrees(np.arctan2(py-last[1],px-last[0]))%360
            ni=np.array([px,py,ps,nd])
            last=ni.copy()
            if self.mu is not None: ni=(ni-self.mu)/self.sigma
            h,c=self.lstm.forward(ni,h,c)
        return preds
