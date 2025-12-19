import os
import json
import shutil
import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Tuple, Dict, Optional, Any, List

SIM_TIME = 3600 * 30

# GLOBAL CONFIGURATION & DEFAULTS

# Numba Check
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    print("System: Numba not found. Simulation running in pure Python. This will be slower.")

@dataclass
class BatteryConfig:
    battery_id: str
    nominal_capacity_Ah: float
    min_voltage_V: float
    max_voltage_V: float
    source_type: str = "generic"
    
    @classmethod
    def from_dict(cls, data: Dict):
        valid_keys = {k for k in cls.__annotations__}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)

class SimulationDefaults:
    """Central repository for simulation constants. If you want to edit their starting values, you can change them here."""
    INITIAL_PARAMS = {
        'Q': 2.0,       'k': 5e-4,      'c': 0.5,
        'R_s': 0.13,     'R1': 0.5,     'C1': 20.0
    }
    
    # [min, max]
    PARAM_BOUNDS = {
        'k': (1e-6, 1e-2), 
        'c': (0.01, 0.99),
        'R_s': (0.001, 1.0), 
        'R1': (0.001, 1.0), 
        'C1': (10.0, 5000.0),
        'Q': (0.1, 100.0) # Will be refined dynamically later based on nominal
    }
    
    ANCHOR_WEIGHTS = {
        'k': 200.0, 'c': 200.0, 'Q': 10.0, 
        'R_s': 10.0, 'R1': 30.0, 'C1': 30.0
    }

# MODEL

class KiBaMPhysics:
    """Contains the raw mathematical solver provided by the professor in the lab plan."""
    @staticmethod
    def solver(dt_vec, I_vec, y1_start, y2_start, k, c, Q_total, R_s, R1, C1, soc_ref, voc_ref):
        n = len(dt_vec)
        y1 = np.zeros(n)
        y2 = np.zeros(n)
        v_model = np.zeros(n)
        v_rc = np.zeros(n)
        soc_model = np.zeros(n)
        
        curr_y1, curr_y2 = y1_start, y2_start
        curr_v_rc = 0.0
        tau = R1 * C1
        
        for i in range(n):
            dt, I = dt_vec[i], I_vec[i]
            
            # KiBaM Equations
            h1, h2 = curr_y1 / c, curr_y2 / (1.0 - c)
            diff_term = k * (h2 - h1)
            curr_y1 += dt * (-I + diff_term)
            curr_y2 += dt * (-diff_term)
            
            # Prevent negative charge & Crash Voltage 
            if curr_y1 < 0:
                curr_y1 = 0 # Tank is empty
            
            if curr_y2 < 0:
                curr_y2 = 0
                
            # RC Circuit
            if tau > 0:
                exp_val = math.exp(-dt / tau)
                curr_v_rc = curr_v_rc * exp_val + R1 * I * (1 - exp_val)
            else: curr_v_rc = 0.0
                
            # SoC Calc
            soc = (curr_y1 + curr_y2) / Q_total
            soc_clamped = min(max(soc, 0.0), 1.0) 
            
            # Voltage Lookup
            v_oc = np.interp(soc_clamped, soc_ref, voc_ref)
            v_term = v_oc - (I * R_s) - curr_v_rc
            
            # If available tank is empty, voltage should collapse. This code addresses that.
            # If we try to draw current (I > 0) from an empty tank (y1 ~ 0), 
            # voltage should drop to 0 immediately.
            # The reason why this code is here is due to the OCV curve projection. It may not be perfect.
            if curr_y1 <= 1e-4 and I > 0:
                v_term = 0.0 # Force cutoff
            
            y1[i], y2[i] = curr_y1, curr_y2
            v_model[i], v_rc[i], soc_model[i] = v_term, curr_v_rc, soc
            
        return y1, y2, v_model, soc_model, v_rc

if HAS_NUMBA:
    KiBaMPhysics.solver = jit(nopython=True)(KiBaMPhysics.solver)

class BatteryModel:
    """Physics Engine."""
    def __init__(self, ocv_df: pd.DataFrame):
        self.ocv_df = ocv_df.sort_values(by='SoC')
        self.soc_vals = self.ocv_df['SoC'].values.astype(np.float64)
        self.voc_vals = (self.ocv_df['Voltage_OCV'].values.astype(np.float64) 
                         if 'Voltage_OCV' in self.ocv_df.columns 
                         else self.ocv_df['Voltage_V'].values.astype(np.float64))

    def run(self, df_input: pd.DataFrame, params: Dict, 
            initial_soc: float = None, cutoff_voltage: float = None) -> pd.DataFrame:
        
        Q_total_Ah = params.get('Q', SimulationDefaults.INITIAL_PARAMS['Q'])
        k = params.get('k', SimulationDefaults.INITIAL_PARAMS['k'])
        c = params.get('c', SimulationDefaults.INITIAL_PARAMS['c'])
        R_s = params.get('R_s', SimulationDefaults.INITIAL_PARAMS['R_s'])
        R1 = params.get('R1', SimulationDefaults.INITIAL_PARAMS['R1'])
        C1 = params.get('C1', SimulationDefaults.INITIAL_PARAMS['C1'])
        Q_total = Q_total_Ah * 3600.0

        if initial_soc is not None:
            start_soc = initial_soc
        elif 'Voltage_V' in df_input.columns:
            # We have reference data, we can estimate start SoC
            v_start_term = df_input['Voltage_V'].iloc[0]
            i_start = df_input['Current_A'].iloc[0]
            v_start_est = v_start_term + (i_start * R_s)
            start_soc = np.interp(v_start_est, self.voc_vals, self.soc_vals)
            start_soc = max(0.01, min(1.0, start_soc))
        else:
            # We are generating a simulation from scratch. 
            # We'll just assume full SoC
            start_soc = 1.0

        y1_start = start_soc * c * Q_total
        y2_start = start_soc * (1 - c) * Q_total

        dt_vec = df_input['dt'].values.astype(np.float64)
        I_vec = df_input['Current_A'].values.astype(np.float64)

        y1, y2, v_model, soc_model, v_rc = KiBaMPhysics.solver(
            dt_vec, I_vec, y1_start, y2_start, k, c, Q_total, R_s, R1, C1, 
            self.soc_vals, self.voc_vals
        )

        results = pd.DataFrame({
            'Time_s': df_input['Time_s'],
            'Current_A': df_input['Current_A'],
            'V_model': v_model,
            'SoC_model': soc_model,
            'V_rc': v_rc,
            'y1': y1,
            'y2': y2
        })

        if cutoff_voltage is not None:
            cutoff_idx = results.index[results['V_model'] < cutoff_voltage]
            if not cutoff_idx.empty:
                results = results.iloc[:cutoff_idx[0]+1]

        return results

# OPTIMIZER

class ModelOptimizer:
    def __init__(self, model: BatteryModel, target_data: pd.DataFrame, nominal_Q: float):
        self.model = model
        self.df_target = target_data
        self.nominal_Q = nominal_Q
        self.v_real = target_data['Voltage_V'].values
        self.anchor_params: Optional[Dict] = None

    def _calculate_loss(self, params: Dict) -> float:
        try:
            df_sim = self.model.run(self.df_target, params)
            v_model = df_sim['V_model'].values
            mse = np.mean((self.v_real - v_model) ** 2)
            rmse = np.sqrt(mse)
            
            penalty = 0.0
            if self.anchor_params:
                for key, weight in SimulationDefaults.ANCHOR_WEIGHTS.items():
                    ref = self.anchor_params.get(key, 0)
                    if ref > 1e-9:
                        rel_diff = (params[key] - ref) / ref
                        penalty += weight * (rel_diff ** 2)
            return rmse + (penalty * 0.01)
        except: return float('inf')

    def _perturb(self, params: Dict) -> Dict:
        new_p = params.copy()
        # Random Walk
        new_p['k']   += np.random.normal(0, new_p['k']*0.02)
        new_p['c']   += np.random.normal(0, new_p['c']*0.0002)
        new_p['R_s'] += np.random.normal(0, new_p['R_s']*0.10)
        new_p['R1']  += np.random.normal(0, new_p['R1']*0.05)
        new_p['C1']  += np.random.normal(0, new_p['C1']*0.05)
        new_p['Q']   += np.random.normal(0, new_p['Q']*0.01)
        
        # Parameter Bounds
        bounds = SimulationDefaults.PARAM_BOUNDS
        
        for key in ['k', 'c', 'R_s', 'R1', 'C1']:
            low, high = bounds[key]
            new_p[key] = max(low, min(high, new_p[key]))
            
        # Q Logic (Dynamic based on nominal)
        min_Q, max_Q = self.nominal_Q * 0.1, self.nominal_Q * 1.5
        new_p['Q'] = max(min_Q, min(max_Q, new_p['Q']))
        
        return new_p

    def run_annealing(self, start_params: Dict, initial_temp=1.0) -> Dict:
        T = initial_temp
        current_params = start_params.copy()
        current_loss = self._calculate_loss(current_params)
        best_params = current_params.copy()
        best_loss = current_loss
        
        total_accepted = 0
        iterations = 0
        
        while T > 1e-4:
            for _ in range(200): # Steps per temp
                new_params = self._perturb(current_params)
                new_loss = self._calculate_loss(new_params)
                iterations += 1
                
                delta = new_loss - current_loss
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_params = new_params
                    current_loss = new_loss
                    total_accepted += 1
                    if current_loss < best_loss:
                        best_loss = current_loss
                        best_params = current_params.copy()
            T *= 0.95 # Cool down ( Alpha )
            
        # Attach stats to the params dict if needed, or just return
        best_params['_meta_loss'] = best_loss
        best_params['_meta_acc_rate'] = total_accepted / iterations
        return best_params

# DATA & STORAGE

class DataStandardizer:
    @staticmethod
    def standardize(df_raw: pd.DataFrame, config: BatteryConfig, c_type: str) -> pd.DataFrame:
        mapping = {
            'Time': 'Time_s', 'time': 'Time_s', 't': 'Time_s',
            'Voltage': 'Voltage_V', 'voltage': 'Voltage_V', 'V': 'Voltage_V', 'Voltage_measured': 'Voltage_V',
            'Current': 'Current_A', 'current': 'Current_A', 'I': 'Current_A', 'Current_measured': 'Current_A'
        }
        df = df_raw.rename(columns=mapping).copy()
        
        for col in ['Time_s', 'Voltage_V', 'Current_A']:
            if col not in df.columns: df[col] = np.nan

        # Zero Time
        if not df.empty and 'Time_s' in df.columns:
             df['Time_s'] -= df['Time_s'].iloc[0]

        # Rolling average for smoothing
        if 'Voltage_V' in df.columns and len(df) > 10:
            df['Voltage_V'] = df['Voltage_V'].rolling(window=7, min_periods=1, center=True).mean()

        # Sign Convention (Discharge = Positive Current Flow)
        if c_type == 'discharge' and df['Current_A'].mean() < 0:
            df['Current_A'] *= -1
        elif c_type == 'charge' and df['Current_A'].mean() > 0:
            df['Current_A'] *= -1

        df['dt'] = df['Time_s'].diff().fillna(0)
        
        q_cumsum = (df['Current_A'] * df['dt']).cumsum() / 3600.0
        start_soc = 1.0 if c_type == 'discharge' else 0.0
        df['SoC_measured'] = start_soc - (q_cumsum / config.nominal_capacity_Ah)
        
        return df

class NasaIngestor:
    """Handles .mat file parsing."""
    @staticmethod
    def _recursive_mat_dict(mat_obj):
        if isinstance(mat_obj, scipy.io.matlab.mio5_params.mat_struct):
            return {f: NasaIngestor._recursive_mat_dict(getattr(mat_obj, f)) 
                    for f in mat_obj._fieldnames}
        if isinstance(mat_obj, np.ndarray):
            if mat_obj.dtype == 'object' and mat_obj.size > 0:
                return [NasaIngestor._recursive_mat_dict(elem) for elem in mat_obj.flat]
            return mat_obj
        return mat_obj

    @staticmethod
    def _parse_date(d_vec):
        try:
            d = np.array(d_vec).flatten().astype(int)
            return datetime(d[0], d[1], d[2], d[3], d[4], d[5])
        except: return None

    def process_file(self, file_path: str, config: BatteryConfig):
        if not os.path.exists(file_path): return None, None, None
        print(f"Loading {file_path}...")
        
        try:
            mat = scipy.io.loadmat(file_path, struct_as_record=False, squeeze_me=True)
            keys = [k for k in mat.keys() if not k.startswith('__')]
            if not keys: return None, None, None
            
            raw_data = self._recursive_mat_dict(mat[keys[0]])
            cycles_raw = raw_data.get('cycle', [])
        except Exception as e:
            print(f"MATLAB Load Error: {e}")
            return None, None, None

        aging, imp, cycles = [], [], {}
        print(f"Processing {len(cycles_raw)} cycles...")

        for i, cycle in enumerate(cycles_raw):
            try:
                c_type = str(cycle.get('type', '')).strip()
                ts = self._parse_date(cycle.get('time', []))
                data = cycle.get('data', {})

                if c_type in ['charge', 'discharge']:
                    # Extract Time Series
                    cols = {}
                    for k in ['Time', 'Voltage_measured', 'Current_measured', 'Temperature_measured']:
                        if k in data: cols[k] = data[k]
                    
                    df_raw = pd.DataFrame(cols)
                    if not df_raw.empty:
                        df_std = DataStandardizer.standardize(df_raw, config, c_type)
                        cycles[i] = {'type': c_type, 'date': ts, 'data': df_std}
                        
                        if c_type == 'discharge' and 'Capacity' in data:
                            cap = data['Capacity'] if isinstance(data['Capacity'], (int, float)) else data['Capacity'][0]
                            aging.append({'Cycle': i, 'Capacity': float(cap), 'Date': ts})
                            
                elif c_type == 'impedance':
                    entry = {'Cycle': i, 'Date': ts}
                    if 'Re' in data: entry['Re'] = float(data['Re'])
                    if 'Rct' in data: entry['Rct'] = float(data['Rct'])
                    imp.append(entry)
            except: continue
            
        return pd.DataFrame(aging), pd.DataFrame(imp), cycles

class StorageManager:
    ROOT = os.path.abspath("Batteries")

    def __init__(self):
        if not os.path.exists(self.ROOT): os.makedirs(self.ROOT)

    def _safe_path(self, *parts):
        path = os.path.abspath(os.path.join(self.ROOT, *parts))
        if os.path.commonprefix([path, self.ROOT]) != self.ROOT:
            raise ValueError("Security Path Traversal Error")
        return path

    def create_battery(self, config: BatteryConfig):
        b_dir = self._safe_path(config.battery_id)
        os.makedirs(os.path.join(b_dir, "BatteryData"), exist_ok=True)
        with open(os.path.join(b_dir, "battery_details.json"), 'w') as f:
            json.dump(asdict(config), f, indent=4)
            
    def delete_battery(self, battery_id: str):
        b_dir = self._safe_path(battery_id)
        if os.path.exists(b_dir):
            if input(f"Delete {battery_id}? [y/N]: ").lower() == 'y':
                shutil.rmtree(b_dir)
                print("Deleted.")

    def load_config(self, battery_id: str) -> Optional[BatteryConfig]:
        try:
            path = self._safe_path(battery_id, "battery_details.json")
            if not os.path.exists(path): return None
            with open(path, 'r') as f:
                return BatteryConfig.from_dict(json.load(f))
        except: return None

    def save_cycle_data(self, battery_id: str, df_aging: pd.DataFrame, df_imp: pd.DataFrame, cycles: Dict):
        data_dir = self._safe_path(battery_id, "BatteryData")
        if not df_aging.empty: df_aging.to_csv(os.path.join(data_dir, "aging_summary.csv"), index=False)
        if not df_imp.empty: df_imp.to_csv(os.path.join(data_dir, "impedance_summary.csv"), index=False)
            
        dfs = []
        for cid, cdata in cycles.items():
            tmp = cdata['data'].copy()
            tmp['Cycle_ID'] = cid
            tmp['Type'] = cdata['type']
            dfs.append(tmp)
        
        if dfs:
            full_df = pd.concat(dfs)
            full_df.to_csv(os.path.join(data_dir, "cycle_data_all.csv.gz"), index=False, compression='gzip')

    def load_cycle_data(self, battery_id: str):
        data_dir = self._safe_path(battery_id, "BatteryData")
        cycles, df_aging, df_imp = {}, pd.DataFrame(), pd.DataFrame()
        
        p_aging = os.path.join(data_dir, "aging_summary.csv")
        p_imp = os.path.join(data_dir, "impedance_summary.csv")
        p_cyc = os.path.join(data_dir, "cycle_data_all.csv.gz")
        
        if os.path.exists(p_aging): df_aging = pd.read_csv(p_aging)
        if os.path.exists(p_imp): df_imp = pd.read_csv(p_imp)
        if os.path.exists(p_cyc):
            print("Loading dataset...")
            df_full = pd.read_csv(p_cyc, compression='gzip')
            for cid, grp in df_full.groupby('Cycle_ID'):
                cycles[cid] = {
                    'data': grp.drop(columns=['Cycle_ID', 'Type']).reset_index(drop=True),
                    'type': grp['Type'].iloc[0]
                }
        return df_aging, df_imp, cycles

    def get_list(self):
        if not os.path.exists(self.ROOT): return []
        return [d for d in os.listdir(self.ROOT) if os.path.isdir(os.path.join(self.ROOT, d))]

# VISUALIZATION

class Visualizer:
    @staticmethod
    def dashboard(battery_id, df_aging, df_imp, cycles, config):
        fig = plt.figure(figsize=(12, 8))
        gs = fig.add_gridspec(2, 2)
        
        ax1 = fig.add_subplot(gs[0, 0])
        if not df_aging.empty:
            ax1.plot(df_aging['Cycle'], df_aging['Capacity'], 'o-', label='Measured')
        ax1.axhline(config.nominal_capacity_Ah, color='r', linestyle='--', label='Nominal')
        ax1.set_title(f"Capacity Fade ({battery_id})")
        ax1.legend()
        ax1.grid(True, alpha=0.5)
        
        ax2 = fig.add_subplot(gs[0, 1])
        if not df_imp.empty:
            cols = [c for c in ['Re', 'Rct'] if c in df_imp.columns]
            if cols:
                ax2.plot(df_imp['Cycle'], df_imp[cols].sum(axis=1), 'x-', color='orange')
            ax2.set_title("Impedance Evolution")
            ax2.grid(True, alpha=0.5)

        ax3 = fig.add_subplot(gs[1, :])
        dis_cycles = sorted([k for k,v in cycles.items() if v['type'] == 'discharge'], key=int)
        if dis_cycles:
            c_first, c_last = cycles[dis_cycles[0]]['data'], cycles[dis_cycles[-1]]['data']
            ax3.plot(c_first['Time_s'], c_first['Voltage_V'], label='Fresh', color='green')
            ax3.plot(c_last['Time_s'], c_last['Voltage_V'], label='Aged', color='red', linestyle='--')
        ax3.set_title("Voltage Profile Change")
        ax3.legend()
        plt.tight_layout(); plt.show()

    @staticmethod
    def compare_fit(df_real, df_sim, params, title="Fit"):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax1.plot(df_real['Time_s'], df_real['Voltage_V'], color='gray', label='Real')
        ax1.plot(df_sim['Time_s'], df_sim['V_model'], color='red', linestyle='--', label='Model')
        ax1.set_title(f"{title}\nRMSE={params.get('RMSE', 0):.4f}")
        ax1.legend(); ax1.grid(True)
        
        err = df_real['Voltage_V'] - df_sim['V_model']
        ax2.plot(df_sim['Time_s'], err, label='Error')
        ax2.axhline(0, color='k'); ax2.legend()
        plt.tight_layout(); plt.show()

    @staticmethod
    def simulation_result(df_sim, cutoff_v, params=None):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
        
        # Plot 1: Voltage
        ax1.plot(df_sim['Time_s'], df_sim['V_model'], color='blue', label='Voltage')
        ax1.axhline(cutoff_v, color='red', linestyle='--', label='Cutoff')
        ax1.set_ylabel("Voltage (V)")
        ax1.set_title("Simulation Result")
        ax1.legend(loc='upper right')
        ax1.grid(True)
        
        # Plot 2: SoC Mechanics (Normalized)
        # Total SoC is already 0.0 - 1.0
        ax2.plot(df_sim['Time_s'], df_sim['SoC_model'], color='black', linewidth=2, label='Total SoC (y1+y2)')
        
        if 'y1' in df_sim.columns:

            q_est = (df_sim['y1'] + df_sim['y2']).max() # Approximate Q_total in Coulombs/Ah
            
            if q_est > 0:
                y1_norm = df_sim['y1'] / q_est
                ax2.plot(df_sim['Time_s'], y1_norm, color='green', linestyle='--', label='Available (y1) [Normalized]')
        
        ax2.set_ylabel("State of Charge (0.0 - 1.0)")
        ax2.set_xlabel("Time (s)")
        ax2.legend(loc='upper right')
        ax2.grid(True)
        ax2.set_ylim(-0.05, 1.05)
        
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_parameters(df_res):
        """Plots all 6 identified parameters + RMSE."""
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        fig.suptitle("Parameter Evolution", fontsize=16)
        
        # Define grid mapping
        # Row 1: Capacity & Diffusion
        # Row 2: Resistance & Charge
        # Row 3: RC Parameters & Error
        params = [
            ('Q', 'Capacity (Ah)', 'blue', axes[0,0]),
            ('k', 'Diffusion k', 'green', axes[0,1]),
            ('c', 'Geometry c', 'purple', axes[0,2]),
            ('R_s', 'Series Res (Ohms)', 'red', axes[1,0]),
            ('1-c', 'Inactive Fraction', 'magenta', axes[1,1]), 
            ('RMSE', 'Fit Error (V)', 'black', axes[1,2]),
            ('R1', 'RC Res (Ohms)', 'orange', axes[2,0]),
            ('C1', 'RC Cap (F)', 'brown', axes[2,1])
        ]
        
        # Calculate 1-c for visualization
        if 'c' in df_res.columns:
            df_res['1-c'] = 1.0 - df_res['c']

        for col, label, color, ax in params:
            if col in df_res.columns:
                ax.plot(df_res['Cycle'], df_res[col], marker='o', markersize=3, color=color, alpha=0.7)
                ax.set_ylabel(label)
                ax.grid(True, linestyle='--', alpha=0.5)
            else:
                ax.text(0.5, 0.5, f"{col} missing", ha='center')

        axes[2,2].axis('off') # Empty slot
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

# WORKFLOWS
# Some of these aren't really that useful but they were maintained for the sake of posterity.
# They have some, but niche applications.

class LoadProfileGenerator:
    @staticmethod
    def constant_current(current, duration, dt=1.0):
        t = np.arange(0, duration + dt, dt)
        return pd.DataFrame({'Time_s': t, 'Current_A': abs(current), 'dt': dt})

    @staticmethod
    def pulse_discharge(high_A, low_A, t_high, t_low, total_dur, dt=1.0):
        t_arr, i_arr = [], []
        curr_t = 0
        while curr_t < total_dur:
            steps = int(t_high/dt)
            t_arr.extend([curr_t + i*dt for i in range(steps)])
            i_arr.extend([abs(high_A)] * steps)
            curr_t += t_high
            
            steps = int(t_low/dt)
            t_arr.extend([curr_t + i*dt for i in range(steps)])
            i_arr.extend([abs(low_A)] * steps)
            curr_t += t_low
        df = pd.DataFrame({'Time_s': t_arr, 'Current_A': i_arr, 'dt': dt})
        return df[df['Time_s'] <= total_dur]

class SimulationController:
    def __init__(self):
        self.store = StorageManager()
        self.ocv_tools = OCVTools()
        self.ingestor = NasaIngestor()

    def _get_ocv(self, bid):
        path = self.store._safe_path(bid, "BatteryData", "ocv_reference.csv")
        if os.path.exists(path): return pd.read_csv(path)
        print("Error: OCV Reference missing. Extract it first.")
        return None

    def extract_ocv(self, bid):
        _, df_imp, cycles = self.store.load_cycle_data(bid)
        config = self.store.load_config(bid)
        if not config: return
        
        df_ocv = self.ocv_tools.extract_ocv(cycles, df_imp, config)
        if df_ocv is not None:
            path = self.store._safe_path(bid, "BatteryData", "ocv_reference.csv")
            df_ocv.to_csv(path, index=False)
            print("OCV Extracted and Saved.")
            plt.plot(df_ocv['SoC'], df_ocv['Voltage_OCV'])
            plt.title("Extracted OCV"); plt.show()

    def run_lifecycle(self, bid):
        config = self.store.load_config(bid)
        if not config: return
        df_ocv = self._get_ocv(bid)
        if df_ocv is None: return

        _, _, cycles = self.store.load_cycle_data(bid)
        dis_keys = sorted([k for k,v in cycles.items() if v['type'] == 'discharge'], key=int)
        
        model = BatteryModel(df_ocv)
        current_params = SimulationDefaults.INITIAL_PARAMS.copy()
        current_params['Q'] = config.nominal_capacity_Ah
        results = []

        print(f"Lifecycle Analysis: {len(dis_keys)} cycles...")
        for i, cid in enumerate(dis_keys):
            df_cyc = cycles[cid]['data']
            opt = ModelOptimizer(model, df_cyc, config.nominal_capacity_Ah)
            if i > 0: opt.anchor_params = current_params
            
            best_p = opt.run_annealing(current_params, initial_temp=1.0 if i==0 else 0.15)
            
            # Print status
            rmse = best_p.get('_meta_loss', 0)
            acc = best_p.get('_meta_acc_rate', 0)
            print(f"Cycle {cid}: RMSE={rmse:.4f}, Acc={acc*100:.1f}%, Q={best_p['Q']:.2f}, R_s={best_p['R_s']:.6f}, c={best_p['c']:.6f}, R1={best_p['R1']:.6f}, C1={best_p['C1']:.6f}")
            
            res = best_p.copy()
            res.update({'Cycle': cid, 'RMSE': rmse})
            results.append(res)
            current_params = best_p.copy()

        df_res = pd.DataFrame(results)
        path = self.store._safe_path(bid, "BatteryData", "parameter_evolution.csv")
        df_res.to_csv(path, index=False)
        print(f"\nSaved to {path}")
        
        Visualizer.plot_parameters(df_res)

    def fit_single_cycle(self, bid):
        config = self.store.load_config(bid)
        if not config: return
        df_ocv = self._get_ocv(bid)
        if df_ocv is None: return
        
        _, _, cycles = self.store.load_cycle_data(bid)
        keys = sorted([k for k,v in cycles.items() if v['type']=='discharge'], key=int)
        if not keys: return
        
        try:
            cid = int(input(f"Select Cycle {keys}: "))
            if cid not in cycles: return
            
            print("Fitting...")
            df_cyc = cycles[cid]['data']
            model = BatteryModel(df_ocv)
            opt = ModelOptimizer(model, df_cyc, config.nominal_capacity_Ah)
            
            best_p = opt.run_annealing(SimulationDefaults.INITIAL_PARAMS)
            df_sim = model.run(df_cyc, best_p)
            
            print("Identified Params:", best_p)
            Visualizer.compare_fit(df_cyc, df_sim, best_p, title=f"Single Cycle Fit (C{cid})")
        except ValueError: print("Invalid selection.")

    def run_manual_sim(self, bid):
        config = self.store.load_config(bid)
        if not config: return
        df_ocv = self._get_ocv(bid)
        if df_ocv is None: return
        
        _, _, cycles = self.store.load_cycle_data(bid)
        keys = sorted([k for k,v in cycles.items() if v['type']=='discharge'], key=int)
        if not keys: return

        try:
            cid = int(input(f"Select Reference Cycle {keys}: "))
            df_cyc = cycles[cid]['data']
            
            print("Enter Parameters:")
            p = SimulationDefaults.INITIAL_PARAMS.copy()

            p['Q'] = float(input(f"Q (Ah) [{p['Q']}]: ") or p['Q'])
            p['k'] = float(input(f"k [{p['k']}]: ") or p['k'])
            p['c'] = float(input(f"c [{p['c']}]: ") or p['c'])
            p['R_s'] = float(input(f"R_s [{p['R_s']}]: ") or p['R_s'])
            p['R1'] = float(input(f"R1 [{p['R1']}]: ") or p['R1'])
            p['C1'] = float(input(f"C1 [{p['C1']}]: ") or p['C1'])
            
            model = BatteryModel(df_ocv)
            df_sim = model.run(df_cyc, p)

            # Calculate RMSE explicitly for the manual run
            v_real = df_cyc['Voltage_V'].values
            v_model = df_sim['V_model'].values
            # Ensure lengths match (in case of cutoff) before calc
            min_len = min(len(v_real), len(v_model))
            mse = np.mean((v_real[:min_len] - v_model[:min_len]) ** 2)
            p['RMSE'] = np.sqrt(mse)

            Visualizer.compare_fit(df_cyc, df_sim, p, title="Manual Parameter Test")
        except ValueError: print("Invalid input.")

    def playground(self, bid):
        config = self.store.load_config(bid)
        if not config: return
        df_ocv = self._get_ocv(bid)
        if df_ocv is None: return
        
        params = SimulationDefaults.INITIAL_PARAMS.copy()
        params['Q'] = config.nominal_capacity_Ah
        
        print("\n[1] Manual Parameters\n[2] Load from Lifecycle Analysis")
        if input("Choice: ") == '2':
            p_path = self.store._safe_path(bid, "BatteryData", "parameter_evolution.csv")
            if os.path.exists(p_path):
                df_p = pd.read_csv(p_path)
                try:
                    cid = int(input(f"Enter Cycle to mimic ({df_p['Cycle'].min()}-{df_p['Cycle'].max()}): "))
                    row = df_p[df_p['Cycle'] == cid]
                    if not row.empty:
                        params = row.iloc[0].to_dict()
                        print(f"Loaded Params from Cycle {cid}")
                except: print("Using defaults.")
        
        print("\n[1] Constant Discharge\n[2] Pulse Discharge")
        choice = input("Choice: ")
        dur = SIM_TIME
        
        if choice == '1':
            amp = float(input("Amps: "))
            df_prof = LoadProfileGenerator.constant_current(amp, dur)
        else:
            h_A = float(input("High Amps: "))
            h_T = float(input("High Time: "))
            l_A = float(input("Low Amps (0=Rest): "))
            l_T = float(input("Low Time: "))
            df_prof = LoadProfileGenerator.pulse_discharge(h_A, l_A, h_T, l_T, dur)

        model = BatteryModel(df_ocv)
        df_sim = model.run(df_prof, params, cutoff_voltage=config.min_voltage_V)
        print(f"Runtime: {df_sim['Time_s'].max()/60:.1f} min")
        # In SimulationController.playground:
        Visualizer.simulation_result(df_sim, config.min_voltage_V) 
        # The visualizer above infers Q from the dataframe sum, so no extra args strictly needed, 
        # but checking Q_total is safer.

class OCVTools:
    def extract_ocv(self, cycles, df_imp, config):
        dis_keys = [k for k,v in cycles.items() if v['type'] == 'discharge']
        if not dis_keys: return None
        
        c_id = sorted(dis_keys, key=int)[0]
        df = cycles[c_id]['data'].copy()
        
        r_est = 0.0
        if not df_imp.empty and 'Re' in df_imp.columns:
            r_est = df_imp.iloc[0]['Re'] + df_imp.iloc[0].get('Rct', 0)
        
        total_ah = (df['Current_A'] * df['dt']).sum() / 3600.0
        cum_ah = (df['Current_A'] * df['dt']).cumsum() / 3600.0
        df['SoC'] = 1.0 - (cum_ah / total_ah)
        
        df['Voltage_OCV'] = df['Voltage_V'] + (df['Current_A'] * r_est)
        return df[['SoC', 'Voltage_V', 'Voltage_OCV']].sort_values('SoC')

# MAIN UI

def main():
    store = StorageManager()
    sim_ctrl = SimulationController()
    
    while True:
        print("\n--- MOBE Battery Toolkit ---")
        batteries = store.get_list()
        print(f"Batteries: {batteries}")
        print("1. Dashboard (Viz)")
        print("2. Create Battery / Import NASA Data")
        print("3. Extract OCV Curve (Required)")
        print("4. Lifecycle Analysis (Train All)")
        print("5. Fit Single Cycle (Train One)")
        print("6. Manual Simulation (Test Params)")
        print("7. Simulation Playground (Predict)")
        print("8. Delete Battery")
        print("q. Quit")
        
        sel = input("Select: ").lower().strip()
        
        if sel == 'q': break
        
        if sel == '1':
            bid = input("Battery ID: ")
            if bid in batteries:
                df_a, df_i, cyc = store.load_cycle_data(bid)
                conf = store.load_config(bid)
                if conf: Visualizer.dashboard(bid, df_a, df_i, cyc, conf)
                
        elif sel == '2':
            bid = input("New Battery ID: ").strip()
            if not bid: continue
            
            print("--- Configuration ---")
            cap = float(input("Nominal Capacity (Ah) [2.0]: ") or 2.0)
            v_max = float(input("Max Voltage (V) [4.2]: ") or 4.2)
            v_min = float(input("Min Voltage (V) [2.7]: ") or 2.7)
            
            conf = BatteryConfig(bid, cap, v_min, v_max)
            store.create_battery(conf)
            print(f"Created {bid}.")
            
            if input("Import NASA .mat file now? [y/N]: ").lower() == 'y':
                path = input("Full path to .mat file: ").strip('"')
                df_a, df_i, cyc = sim_ctrl.ingestor.process_file(path, conf)
                if df_a is not None and not df_a.empty:
                    store.save_cycle_data(bid, df_a, df_i, cyc)
                    print("Import Successful.")
                else: print("Import Failed.")
                
        elif sel == '3':
            bid = input("Battery ID: ")
            if bid in batteries: sim_ctrl.extract_ocv(bid)

        elif sel == '4':
            bid = input("Battery ID: ")
            if bid in batteries: sim_ctrl.run_lifecycle(bid)
            
        elif sel == '5':
            bid = input("Battery ID: ")
            if bid in batteries: sim_ctrl.fit_single_cycle(bid)

        elif sel == '6':
            bid = input("Battery ID: ")
            if bid in batteries: sim_ctrl.run_manual_sim(bid)
            
        elif sel == '7':
            bid = input("Battery ID: ")
            if bid in batteries: sim_ctrl.playground(bid)
            
        elif sel == '8':
            bid = input("Battery ID: ")
            store.delete_battery(bid)

if __name__ == "__main__":
    main()