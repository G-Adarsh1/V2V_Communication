"""
Live Dashboard — Tkinter window that runs in a separate thread
alongside SUMO-GUI showing all real-time metrics.
"""
import tkinter as tk
from tkinter import ttk
import threading, queue, time
from collections import deque

class Dashboard:
    def __init__(self):
        self.q      = queue.Queue()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        time.sleep(0.5)   # let window init

    def _run(self):
        self.root = tk.Tk()
        self.root.title("V2V Intelligence Dashboard")
        self.root.geometry("480x720")
        self.root.configure(bg='#050f05')
        self.root.resizable(False, False)
        self._build_ui()
        self._poll()
        self.root.mainloop()

    def _label(self, parent, text, fg='#00ff88', font_size=10, **kw):
        return tk.Label(parent, text=text, fg=fg, bg='#050f05',
                        font=('Courier New', font_size, 'bold'), **kw)

    def _build_ui(self):
        r = self.root
        # Title
        tk.Label(r, text="V2V COLLISION PREDICTION SYSTEM",
                 fg='#00ff88', bg='#050f05',
                 font=('Courier New',11,'bold')).pack(pady=(10,2))
        tk.Label(r, text="Emergency-Focused V2V Intelligence",
                 fg='#336633', bg='#050f05',
                 font=('Courier New',9)).pack(pady=(0,8))

        # Phase banner
        self.phase_var = tk.StringVar(value="⏳ INITIALIZING...")
        tk.Label(r, textvariable=self.phase_var,
                 fg='#ffdd00', bg='#0a1a0a',
                 font=('Courier New',10,'bold'),
                 relief='groove', padx=10, pady=5).pack(fill='x', padx=10)

        # Emergency priority banner (hidden until activated)
        self.emg_banner_var = tk.StringVar(value="")
        self.emg_banner_lbl = tk.Label(r, textvariable=self.emg_banner_var,
                 fg='#ffffff', bg='#cc0044',
                 font=('Courier New',10,'bold'),
                 relief='groove', padx=10, pady=6)

        # ── Risk counts ──────────────────────────────
        frisk = tk.LabelFrame(r, text=" RISK STATUS ", fg='#00ff88',
                              bg='#050f05', font=('Courier New',9,'bold'))
        frisk.pack(fill='x', padx=10, pady=6)
        row = tk.Frame(frisk, bg='#050f05'); row.pack(fill='x', padx=8, pady=4)
        self.safe_var  = tk.StringVar(value="0")
        self.risk_var  = tk.StringVar(value="0")
        self.coll_var  = tk.StringVar(value="0")
        for txt, var, col in [("✅ SAFE","0",'#00ff88'),
                               ("⚠ RISK","0",'#ffaa00'),
                               ("🔴 COLLISION","0",'#ff2244')]:
            f = tk.Frame(row, bg='#0a1a0a', relief='ridge', bd=1)
            f.pack(side='left', expand=True, fill='both', padx=3, pady=2)
            sv = tk.StringVar(value="0")
            if 'SAFE' in txt:   self.safe_var = sv
            elif 'RISK' in txt: self.risk_var = sv
            else:               self.coll_var = sv
            tk.Label(f,text=txt,fg=col,bg='#0a1a0a',
                     font=('Courier New',8,'bold')).pack()
            tk.Label(f,textvariable=sv,fg=col,bg='#0a1a0a',
                     font=('Courier New',20,'bold')).pack()

        # ── ML metrics ───────────────────────────────
        fml = tk.LabelFrame(r, text=" ML MODEL METRICS ",
                            fg='#00ccff', bg='#050f05',
                            font=('Courier New',9,'bold'))
        fml.pack(fill='x', padx=10, pady=4)
        self.auc_var  = tk.StringVar(value="—")
        self.r2_var   = tk.StringVar(value="—")
        self.acc_var  = tk.StringVar(value="—")
        for lbl, var, col in [("RF  AUC ",self.auc_var,'#00ccff'),
                               ("RF  ACC ",self.acc_var,'#00ccff'),
                               ("LSTM R² ",self.r2_var, '#88ffcc')]:
            row2 = tk.Frame(fml, bg='#050f05'); row2.pack(fill='x', padx=8, pady=1)
            tk.Label(row2,text=lbl,fg='#336655',bg='#050f05',
                     font=('Courier New',9,'bold'),width=10,anchor='w').pack(side='left')
            tk.Label(row2,textvariable=var,fg=col,bg='#050f05',
                     font=('Courier New',9,'bold')).pack(side='left')

        # ── Network metrics ───────────────────────────
        fnet = tk.LabelFrame(r, text=" NETWORK METRICS ",
                             fg='#ffaa00', bg='#050f05',
                             font=('Courier New',9,'bold'))
        fnet.pack(fill='x', padx=10, pady=4)
        self.pdr_var   = tk.StringVar(value="—")
        self.delay_var = tk.StringVar(value="—")
        self.hop_var   = tk.StringVar(value="—")
        self.beacon_var= tk.StringVar(value="—")
        for lbl, var in [("Avg PDR      ", self.pdr_var),
                         ("Avg Delay ms ", self.delay_var),
                         ("Last Hop Cnt ", self.hop_var),
                         ("Beacons Sent ", self.beacon_var)]:
            row3 = tk.Frame(fnet, bg='#050f05'); row3.pack(fill='x',padx=8,pady=1)
            tk.Label(row3,text=lbl,fg='#665533',bg='#050f05',
                     font=('Courier New',9,'bold'),width=14,anchor='w').pack(side='left')
            tk.Label(row3,textvariable=var,fg='#ffaa00',bg='#050f05',
                     font=('Courier New',9,'bold')).pack(side='left')

        # ── Vehicles saved / collisions prevented ────
        fsave = tk.LabelFrame(r, text=" IMPACT REPORT ",
                              fg='#88ff44', bg='#050f05',
                              font=('Courier New',9,'bold'))
        fsave.pack(fill='x', padx=10, pady=4)
        self.saved_var   = tk.StringVar(value="0")
        self.prevented_var=tk.StringVar(value="0")
        self.emg_time_var = tk.StringVar(value="—")
        self.emg_vehicles_var = tk.StringVar(value="—")
        for lbl, var, col in [("Vehicles Saved    ",self.saved_var,'#88ff44'),
                               ("Collisions Prevent",self.prevented_var,'#88ff44'),
                               ("Emg Response Time ",self.emg_time_var,'#ffdd00'),
                               ("Emergency Vehicles",self.emg_vehicles_var,'#ff88cc')]:
            row4 = tk.Frame(fsave, bg='#050f05'); row4.pack(fill='x',padx=8,pady=1)
            tk.Label(row4,text=lbl,fg='#446633',bg='#050f05',
                     font=('Courier New',9,'bold'),width=20,anchor='w').pack(side='left')
            tk.Label(row4,textvariable=var,fg=col,bg='#050f05',
                     font=('Courier New',9,'bold')).pack(side='left')

        # ── Alert log ─────────────────────────────────
        flog = tk.LabelFrame(r, text=" LIVE ALERT LOG ",
                             fg='#ff6644', bg='#050f05',
                             font=('Courier New',9,'bold'))
        flog.pack(fill='both', expand=True, padx=10, pady=4)
        self.log_text = tk.Text(flog, height=10, bg='#030a03', fg='#ff8866',
                                font=('Courier New',8), state='disabled',
                                relief='flat', insertbackground='#00ff88')
        sb = tk.Scrollbar(flog, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=sb.set)
        sb.pack(side='right', fill='y')
        self.log_text.pack(fill='both', expand=True, padx=4, pady=4)

        # tag colours
        self.log_text.tag_config('COLLISION', foreground='#ff2244')
        self.log_text.tag_config('RISK',      foreground='#ffaa00')
        self.log_text.tag_config('EMERGENCY', foreground='#ff6600')
        self.log_text.tag_config('SAFE',      foreground='#00ff88')
        self.log_text.tag_config('INFO',      foreground='#00ccff')

    def _poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                self._handle(msg)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _handle(self, msg):
        t = msg.get('type')
        if t == 'risk':
            self.safe_var.set(str(msg['safe']))
            self.risk_var.set(str(msg['risk']))
            self.coll_var.set(str(msg['collision']))
        elif t == 'ml':
            self.auc_var.set(f"{msg['auc']:.4f}")
            self.acc_var.set(f"{msg['acc']:.4f}")
            self.r2_var.set(f"{msg['r2']:.4f}")
        elif t == 'net':
            self.pdr_var.set(f"{msg['pdr']:.4f}")
            self.delay_var.set(f"{msg['delay']:.2f} ms")
            self.hop_var.set(str(msg.get('hop',0)))
            self.beacon_var.set(str(msg.get('beacons',0)))
        elif t == 'impact':
            self.saved_var.set(str(msg['saved']))
            self.prevented_var.set(str(msg['prevented']))
            if msg.get('emg_time'):
                self.emg_time_var.set(f"{msg['emg_time']:.1f} s")
            if msg.get('emg_vehicles'):
                self.emg_vehicles_var.set(msg['emg_vehicles'])
        elif t == 'emg_priority':
            if msg.get('active'):
                self.emg_banner_var.set("🚑  EMERGENCY PRIORITY ACTIVE  🚒")
                self.emg_banner_lbl.pack(fill='x', padx=10, pady=(0,4))
            else:
                self.emg_banner_var.set("")
                self.emg_banner_lbl.pack_forget()
        elif t == 'phase':
            self.phase_var.set(msg['text'])
        elif t == 'log':
            self._append_log(msg['text'], msg.get('level','INFO'))

    def _append_log(self, text, level='INFO'):
        ts = time.strftime('%H:%M:%S')
        self.log_text.configure(state='normal')
        self.log_text.insert('end', f"[{ts}] {text}\n", level)
        self.log_text.see('end')
        self.log_text.configure(state='disabled')

    # ── public API (thread-safe) ──────────────────────────────
    def update_risk(self, safe, risk, collision):
        self.q.put({'type':'risk','safe':safe,'risk':risk,'collision':collision})
    def update_ml(self, auc, acc, r2):
        self.q.put({'type':'ml','auc':auc,'acc':acc,'r2':r2})
    def update_net(self, pdr, delay, hop=0, beacons=0):
        self.q.put({'type':'net','pdr':pdr,'delay':delay,'hop':hop,'beacons':beacons})
    def update_impact(self, saved, prevented, emg_time=None, emg_vehicles=None):
        self.q.put({'type':'impact','saved':saved,'prevented':prevented,
                    'emg_time':emg_time,'emg_vehicles':emg_vehicles})

    def set_emergency_priority(self, active: bool):
        self.q.put({'type':'emg_priority','active':active})
    def set_phase(self, text):
        self.q.put({'type':'phase','text':text})
    def log(self, text, level='INFO'):
        self.q.put({'type':'log','text':text,'level':level})