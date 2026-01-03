"""
MOBE Battery Toolkit
"""

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

simTime = 3600 * 30
userTauDivision = 4


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
        'R_s': 0.1,     'R1': 0.1,     'C1': 20.0
    }
    
    # [min, max]
    PARAM_BOUNDS = {
        'k': (0.0001, 0.01),
        'c': (0.40, 0.60),  
        'R_s': (0.070, 0.5),
        'R1': (0.070, 0.5),
        'C1': (10.0, 100.0), 
        'Q': (0.1, 100.0)
    }
    
    ANCHOR_WEIGHTS = {
        'k': 200.0, 'c': 200.0, 'Q': 10.0, 
        'R_s': 10.0, 'R1': 30.0, 'C1': 30.0
    }

    PERTURBATION_SCALES = {
        'k': 0.01,   'c': 0.005,  'Q': 0.005,
        'R_s': 0.03, 'R1': 0.02,  'C1': 0.02
    }

    ANNEALING_STEPS = 200

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
            'Time_s': df_input['Time_s'].values,
            'Current_A': df_input['Current_A'].values,
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
        self.cycle_number = 0  # Track cycle number for adaptive anchoring

    def _calculate_loss(self, params: Dict) -> float:
        try:
            df_sim = self.model.run(self.df_target, params)
            v_model = df_sim['V_model'].values
            
            # Ensure arrays have same length before calculating MSE
            min_len = min(len(self.v_real), len(v_model))
            mse = np.mean((self.v_real[:min_len] - v_model[:min_len]) ** 2)
            rmse = np.sqrt(mse)
            
            penalty = 0.0
            if self.anchor_params:
                # Adaptive anchor strength - increase with cycle number
                anchor_multiplier = 1.0 + (self.cycle_number / 100) * 0.5  # 50% stronger by cycle 100
                
                for key, weight in SimulationDefaults.ANCHOR_WEIGHTS.items():
                    ref = self.anchor_params.get(key, 0)
                    if ref > 1e-9:
                        rel_diff = (params[key] - ref) / ref
                        penalty += (weight * anchor_multiplier) * (rel_diff ** 2)
            return rmse + (penalty * 0.01)
        except Exception as e:
            print(f"Warning: Loss calculation failed: {e}")
            return float('inf')

    def _perturb(self, params: Dict) -> Dict:
        new_p = params.copy()
        
        # Random Walk using dynamic scales
        # We loop through keys to ensure we catch all physics parameters
        for key in ['k', 'c', 'R_s', 'R1', 'C1', 'Q']:
            if key in new_p:
                # Get scale from defaults, fallback to 2% if missing
                scale = SimulationDefaults.PERTURBATION_SCALES.get(key, 0.02)
                
                # Apply Gaussian noise: New = Old + Normal(0, Old * scale)
                new_p[key] += np.random.normal(0, new_p[key] * scale)
        
        # Parameter Bounds
        bounds = SimulationDefaults.PARAM_BOUNDS
        
        for key in ['k', 'c', 'R_s', 'R1', 'C1']:
            low, high = bounds[key]
            new_p[key] = max(low, min(high, new_p[key]))
            
        # Q Logic (Dynamic based on nominal)
        min_Q, max_Q = self.nominal_Q * 0.1, self.nominal_Q * 1.5
        new_p['Q'] = max(min_Q, min(max_Q, new_p['Q']))
        
        return new_p

    def run_annealing(self, start_params: Dict, initial_temp=1.0, cycle_number: int = 0) -> Dict:
        self.cycle_number = cycle_number  # Store for use in _calculate_loss
        
        T = initial_temp
        current_params = start_params.copy()
        current_loss = self._calculate_loss(current_params)
        best_params = current_params.copy()
        best_loss = current_loss
        
        total_accepted = 0
        iterations = 0
        no_improvement_count = 0

        steps_per_temp = getattr(SimulationDefaults, 'ANNEALING_STEPS', 200)
        
        while T > 1e-4 and no_improvement_count < 5:  # Early stopping
            accepted_this_temp = 0
            
            for _ in range(steps_per_temp):  # Steps per temp
                new_params = self._perturb(current_params)
                new_loss = self._calculate_loss(new_params)
                iterations += 1
                
                delta = new_loss - current_loss
                if delta < 0 or random.random() < math.exp(-delta / T):
                    current_params = new_params
                    current_loss = new_loss
                    total_accepted += 1
                    accepted_this_temp += 1
                    
                    if current_loss < best_loss:
                        best_loss = current_loss
                        best_params = current_params.copy()
                        no_improvement_count = 0  # Reset counter
            
            # Adaptive cooling based on acceptance rate
            if accepted_this_temp > 40:  # High acceptance - cool slower
                T *= 0.97
            elif accepted_this_temp < 10:  # Low acceptance - cool faster
                T *= 0.92
            else:
                T *= 0.95
                
            if accepted_this_temp == 0:
                no_improvement_count += 1
        
        # Attach stats to the params dict if needed, or just return
        best_params['_meta_loss'] = best_loss
        best_params['_meta_acc_rate'] = total_accepted / iterations if iterations > 0 else 0
        return best_params

# DATA & STORAGE

class DataStandardizer:
    @staticmethod
    def standardize(df_raw: pd.DataFrame, config, cycle_type: str) -> pd.DataFrame:
        """
        Convert raw NASA data to standardized format.
        """
        df = df_raw.copy()
        
        # Rename columns to standard format
        df = df.rename(columns={
            'Time': 'Time_s',
            'Voltage': 'Voltage_V',
            'Current': 'Current_A',
            'Temperature': 'Temperature'
        })
        
        # Ensure we have the required columns
        if 'Time_s' not in df.columns or 'Voltage_V' not in df.columns or 'Current_A' not in df.columns:
            raise ValueError("Missing required columns in raw data")
        
        # Calculate time differences
        df['dt'] = df['Time_s'].diff().fillna(0)
        df.loc[df['dt'] < 0, 'dt'] = 0
        
        # NASA data uses positive current for discharge
        if cycle_type in ['discharge', 'pulsed']:
            # For discharge cycles, current should be positive
            df['Current_A'] = df['Current_A'].abs()
        
        # Calculate SoC via coulomb counting
        # Total capacity (Ah) = integral of current over time
        total_capacity_Ah = (df['Current_A'] * df['dt']).sum() / 3600.0
        
        # If we didn't discharge much (e.g., mostly rest), use nominal capacity
        if total_capacity_Ah < config.nominal_capacity_Ah * 0.05:
            print(f"Warning: Very low discharge detected ({total_capacity_Ah:.3f}Ah). Using nominal capacity.")
            total_capacity_Ah = config.nominal_capacity_Ah
        
        # Cumulative Ah discharged
        cumulative_Ah = (df['Current_A'] * df['dt']).cumsum() / 3600.0
        
        # SoC = 1 - (discharged / total)
        df['SoC_measured'] = 1.0 - (cumulative_Ah / total_capacity_Ah)
        df['SoC_measured'] = df['SoC_measured'].clip(0, 1)  # Keep in [0,1]
        
        # Ensure Temperature column exists
        if 'Temperature' not in df.columns:
            df['Temperature'] = 25.0  # Default room temperature
        
        return df[['Time_s', 'Voltage_V', 'Current_A', 'Temperature', 'dt', 'SoC_measured']]

class NasaIngestor:
    """
    Handles importing NASA Battery Data from .mat files.
    Auto-detects between Standard (B0005) and Random Walk (RW3) formats.
    """
    def __init__(self):
        pass

    @staticmethod
    def _recursive_mat_dict(mat_obj):
        """Recursively convert MATLAB objects to Python dicts"""
        if isinstance(mat_obj, dict):
            return {k: NasaIngestor._recursive_mat_dict(v) for k, v in mat_obj.items()}
        elif hasattr(mat_obj, '_fieldnames'):
            return {f: NasaIngestor._recursive_mat_dict(getattr(mat_obj, f)) 
                   for f in mat_obj._fieldnames}
        elif isinstance(mat_obj, np.ndarray):
            if mat_obj.dtype == 'object' and mat_obj.size > 0:
                return [NasaIngestor._recursive_mat_dict(elem) for elem in mat_obj.flat]
            elif mat_obj.size == 1:
                return mat_obj.flat[0]
            return mat_obj
        else:
            return mat_obj

    @staticmethod
    def _parse_date(d_vec):
        try:
            d = np.array(d_vec).flatten().astype(int)
            if len(d) >= 6:
                return datetime(d[0], d[1], d[2], d[3], d[4], d[5])
        except:
            pass
        return None

    def process_file(self, file_path: str, config: BatteryConfig):
        """
        Main entry point for importing a .mat file.
        Returns: (df_aging, df_impedance, cycles_dict)
        """
        if not os.path.exists(file_path):
            print(f"Error: File not found {file_path}")
            return None, None, None

        print(f"Loading {file_path}...")
        try:
            mat = scipy.io.loadmat(file_path, struct_as_record=False, squeeze_me=True)
        except Exception as e:
            print(f"MATLAB load error: {e}")
            return None, None, None

        # Identify data structure
        keys = [k for k in mat.keys() if not k.startswith('__')]
        if not keys:
            print("Error: Empty .mat file")
            return None, None, None
        
        main_key = keys[0]
        root_data = self._recursive_mat_dict(mat[main_key])
        
        # Check format type
        if 'cycle' in root_data:
            print("Detected Standard NASA format (B0005-style)")
            return self._process_standard_format(root_data, config)
        elif 'step' in root_data:
            print("Detected Random Walk format (RW9-style)")
            return self._process_rw_format(root_data, config)
        else:
            print(f"Unknown .mat structure. Keys found: {list(root_data.keys())}")
            return None, None, None

    def _process_standard_format(self, root_data, config):
        """Standard B0005-style processing"""
        raw_cycles = root_data['cycle']
        if not isinstance(raw_cycles, list): raw_cycles = [raw_cycles]
        
        aging_data, impedance_data, cycles_dict = [], [], {}
        print(f"Processing {len(raw_cycles)} standard cycles...")
        
        for i, cycle in enumerate(raw_cycles):
            try:
                c_type = str(cycle.get('type', '')).strip().lower()
                ts = self._parse_date(cycle.get('time', []))
                data = cycle.get('data', {})
                
                if c_type in ['charge', 'discharge']:
                    v = data.get('Voltage_measured')
                    i_curr = data.get('Current_measured')
                    t = data.get('Time')
                    temp = data.get('Temperature_measured')
                    
                    if v is None or i_curr is None or t is None: continue
                    
                    df_raw = pd.DataFrame({'Time': t, 'Voltage': v, 'Current': i_curr, 'Temperature': temp})
                    df_std = DataStandardizer.standardize(df_raw, config, c_type)
                    cycles_dict[i] = {'type': c_type, 'date': ts, 'data': df_std}
                    
                    if c_type == 'discharge' and 'Capacity' in data:
                        cap = data['Capacity']
                        if isinstance(cap, np.ndarray): cap = cap.flat[0]
                        aging_data.append({'Cycle': i, 'Capacity': float(cap), 'Date': ts})
                
                elif c_type == 'impedance':
                    entry = {'Cycle': i, 'Date': ts}
                    if 'Re' in data: entry['Re'] = float(data['Re'])
                    if 'Rct' in data: entry['Rct'] = float(data['Rct'])
                    impedance_data.append(entry)
                    
            except Exception as e: continue
        
        return pd.DataFrame(aging_data), pd.DataFrame(impedance_data), cycles_dict

    def _identify_rw_cycle_type(self, comment: str, step_type: str) -> str:
        """
        Identification for RW datasets.
        Returns: 'low_current', 'reference_discharge', 'pulsed_discharge', 'random_walk', 'rest', or 'charge'
        """
        comment_lower = comment.lower()
        
        # Low current discharge - for OCV extraction
        if 'low current discharge' in comment_lower and '0.04a' in comment_lower:
            return 'low_current'
        
        # Reference discharge - Standard capacity test
        if 'reference discharge' in comment_lower and 'rest' not in comment_lower:
            return 'reference_discharge'
        
        # Pulsed discharge steps
        if 'pulsed load' in comment_lower:
            if 'discharge' in comment_lower:
                return 'pulsed_discharge'
            elif 'rest' in comment_lower:
                return 'pulsed_rest'
        
        # Random walk discharge
        if 'random' in comment_lower or 'walk' in comment_lower:
            return 'random_walk'
        
        # Post-cycle rest periods
        if 'rest post' in comment_lower:
            return 'post_rest'
        
        # Charge steps
        if step_type == 'C' or 'charge' in comment_lower:
            return 'charge'
        
        # Generic discharge
        if step_type == 'D' or 'discharge' in comment_lower:
            return 'discharge'
        
        # Generic rest
        if step_type == 'R' or 'rest' in comment_lower:
            return 'rest'
        
        return 'unknown'

    def _process_rw_format(self, root_data, config):
        """
        RW Logic with cycle type identification:
        Identifies low current, reference, and pulsed discharge cycles separately
        Properly merges pulsed D-R-D-R sequences
        Uses Charge steps as delimiters
        Provides detailed cycle descriptions
        """
        raw_steps = root_data['step']
        if not isinstance(raw_steps, list): raw_steps = [raw_steps]
        
        cycles_dict = {}
        current_segment = []  # Buffer for stitching steps
        segment_id = 1
        pulsed_sequence = []  # Special buffer for pulsed sequences
        in_pulsed_sequence = False
        
        print(f"Analyzing {len(raw_steps)} steps...")
        
        # Track cycle types for summary
        cycle_type_counts = {}
        
        for i, step in enumerate(raw_steps):
            # Extract Type and Comment
            step_type = str(step.get('type', '')).strip().upper()  # C, D, R
            comment = str(step.get('comment', ''))
            
            # Identify cycle type
            cycle_type = self._identify_rw_cycle_type(comment, step_type)
            
            # Helper to extract and validate data
            def ensure_array(val):
                if val is None: return None
                if np.isscalar(val): return np.array([val])
                if isinstance(val, (list, tuple)): return np.array(val)
                if isinstance(val, np.ndarray): return val.flatten() if val.size > 0 else None
                return None
            
            # HANDLE PULSED SEQUENCES
            if cycle_type in ['pulsed_discharge', 'pulsed_rest']:
                in_pulsed_sequence = True
                
                # Extract data
                v = ensure_array(step.get('voltage'))
                i_curr = ensure_array(step.get('current'))
                t = ensure_array(step.get('relativeTime'))
                temp = ensure_array(step.get('temperature'))
                
                if v is not None and i_curr is not None and t is not None:
                    min_len = min(len(v), len(i_curr), len(t))
                    if min_len >= 1:
                        step_df = pd.DataFrame({
                            'Time': t[:min_len],
                            'Voltage': v[:min_len],
                            'Current': i_curr[:min_len],
                            'Temperature': temp[:min_len] if temp is not None and len(temp) >= min_len else np.full(min_len, 25.0)
                        })
                        pulsed_sequence.append({'data': step_df, 'tag': cycle_type, 'comment': comment.lower()})
                continue
            
            # If we were in a pulsed sequence and hit something else, finalize it
            elif in_pulsed_sequence and cycle_type == 'post_rest':
                # Finalize pulsed sequence
                if pulsed_sequence:
                    merged = self._finalize_pulsed_sequence(pulsed_sequence, config)
                    if merged is not None:
                        cycles_dict[segment_id] = merged
                        cycle_type_counts['pulsed'] = cycle_type_counts.get('pulsed', 0) + 1
                        segment_id += 1
                    pulsed_sequence = []
                in_pulsed_sequence = False
                continue
            
            # HANDLE LOW CURRENT DISCHARGE (single step, high value for OCV)
            elif cycle_type == 'low_current':
                v = ensure_array(step.get('voltage'))
                i_curr = ensure_array(step.get('current'))
                t = ensure_array(step.get('relativeTime'))
                temp = ensure_array(step.get('temperature'))
                
                if v is not None and i_curr is not None and t is not None:
                    min_len = min(len(v), len(i_curr), len(t))
                    if min_len >= 1:
                        step_df = pd.DataFrame({
                            'Time': t[:min_len],
                            'Voltage': v[:min_len],
                            'Current': i_curr[:min_len],
                            'Temperature': temp[:min_len] if temp is not None and len(temp) >= min_len else np.full(min_len, 25.0)
                        })
                        
                        df_std = DataStandardizer.standardize(step_df, config, 'discharge')
                        cycles_dict[segment_id] = {
                            'type': 'low_current',
                            'data': df_std,
                            'date': None,
                            'description': 'Low current discharge at 0.04A (OCV reference)'
                        }
                        cycle_type_counts['low_current'] = cycle_type_counts.get('low_current', 0) + 1
                        segment_id += 1
                continue
            
            # HANDLE REFERENCE DISCHARGE
            elif cycle_type == 'reference_discharge':
                v = ensure_array(step.get('voltage'))
                i_curr = ensure_array(step.get('current'))
                t = ensure_array(step.get('relativeTime'))
                temp = ensure_array(step.get('temperature'))
                
                if v is not None and i_curr is not None and t is not None:
                    min_len = min(len(v), len(i_curr), len(t))
                    if min_len >= 1:
                        step_df = pd.DataFrame({
                            'Time': t[:min_len],
                            'Voltage': v[:min_len],
                            'Current': i_curr[:min_len],
                            'Temperature': temp[:min_len] if temp is not None and len(temp) >= min_len else np.full(min_len, 25.0)
                        })
                        
                        df_std = DataStandardizer.standardize(step_df, config, 'discharge')
                        cycles_dict[segment_id] = {
                            'type': 'discharge',
                            'data': df_std,
                            'date': None,
                            'description': 'Reference discharge at 2A'
                        }
                        cycle_type_counts['reference'] = cycle_type_counts.get('reference', 0) + 1
                        segment_id += 1
                continue
            
            # HANDLE CHARGE (delimiter - finalize any current segment)
            elif cycle_type == 'charge':
                # Finalize any pulsed sequence
                if pulsed_sequence:
                    merged = self._finalize_pulsed_sequence(pulsed_sequence, config)
                    if merged is not None:
                        cycles_dict[segment_id] = merged
                        cycle_type_counts['pulsed'] = cycle_type_counts.get('pulsed', 0) + 1
                        segment_id += 1
                    pulsed_sequence = []
                    in_pulsed_sequence = False
                
                # Finalize any regular segment
                if current_segment:
                    merged = self._finalize_segment(current_segment, config)
                    cycles_dict[segment_id] = merged
                    seg_type = merged['type']
                    cycle_type_counts[seg_type] = cycle_type_counts.get(seg_type, 0) + 1
                    segment_id += 1
                    current_segment = []
                continue
            
            # ACCUMULATE OTHER DISCHARGE/REST STEPS
            elif cycle_type in ['discharge', 'rest', 'random_walk']:
                v = ensure_array(step.get('voltage'))
                i_curr = ensure_array(step.get('current'))
                t = ensure_array(step.get('relativeTime'))
                temp = ensure_array(step.get('temperature'))
                
                if v is not None and i_curr is not None and t is not None:
                    min_len = min(len(v), len(i_curr), len(t))
                    if min_len >= 1:
                        step_df = pd.DataFrame({
                            'Time': t[:min_len],
                            'Voltage': v[:min_len],
                            'Current': i_curr[:min_len],
                            'Temperature': temp[:min_len] if temp is not None and len(temp) >= min_len else np.full(min_len, 25.0)
                        })
                        
                        tag = 'discharge' if cycle_type in ['discharge', 'random_walk'] else 'rest'
                        current_segment.append({'data': step_df, 'tag': tag, 'comment': comment.lower()})

        # Finalize any remaining buffers
        if pulsed_sequence:
            merged = self._finalize_pulsed_sequence(pulsed_sequence, config)
            if merged is not None:
                cycles_dict[segment_id] = merged
                cycle_type_counts['pulsed'] = cycle_type_counts.get('pulsed', 0) + 1
                segment_id += 1
        
        if current_segment:
            merged = self._finalize_segment(current_segment, config)
            cycles_dict[segment_id] = merged
            seg_type = merged['type']
            cycle_type_counts[seg_type] = cycle_type_counts.get(seg_type, 0) + 1

        # Print summary
        print(f"\nExtraction Complete. Created {len(cycles_dict)} cycles:")
        for ctype, count in sorted(cycle_type_counts.items()):
            print(f"  - {ctype}: {count} cycles")
        
        # Return empty aging/impedance (RW data usually doesn't have this summary)
        return pd.DataFrame(), pd.DataFrame(), cycles_dict
    
    def _finalize_pulsed_sequence(self, pulsed_steps, config):
        """
        Merges pulsed discharge and rest steps into a single continuous cycle.
        This handles the D-R-D-R-D-R pattern specific to pulsed discharges.
        """
        if not pulsed_steps:
            return None
        
        dfs = []
        cumulative_time = 0.0
        
        print(f"\n  Merging {len(pulsed_steps)} pulsed steps:")
        
        for item in pulsed_steps:
            df = item['data'].copy()
            tag = item['tag']
            
            # Make time continuous
            df['Time'] = df['Time'] + cumulative_time
            
            # Update cumulative time
            if not df.empty:
                step_duration = df['Time'].iloc[-1] - df['Time'].iloc[0]
                cumulative_time = df['Time'].iloc[-1] + 1.0  # 1 second gap
                
                # Print step info
                current_range = f"[{df['Current'].min():.3f}, {df['Current'].max():.3f}]A"
                print(f"    {tag}: {len(df)} points, Duration: {step_duration:.1f}s, Current: {current_range}")
            
            dfs.append(df)
        
        if not dfs:
            return None
        
        # Concatenate all steps
        full_df = pd.concat(dfs, ignore_index=True)
        
        # Standardize
        df_std = DataStandardizer.standardize(full_df, config, 'pulsed')
        
        # Validation
        current_std = df_std['Current_A'].std()
        soc_change = abs(df_std['SoC_measured'].iloc[-1] - df_std['SoC_measured'].iloc[0])
        
        print(f"  Final: {len(df_std)} points, Current std: {current_std:.3f}A, SoC change: {soc_change:.3f}")
        
        if current_std < 0.01:
            print(f"  WARNING: No current variation detected!")
        
        return {
            'type': 'pulsed',
            'data': df_std,
            'date': None,
            'description': f'Pulsed discharge sequence ({len(pulsed_steps)} steps merged)'
        }

    def _finalize_segment(self, segment_list, config):
        """Merges a list of step dataframes into one continuous cycle."""
        
        # Determine Cycle Type (Reference vs Random Walk)
        # If the segment contains "random" or "pulsed", treat as Pulsed/RW.
        # Otherwise, if it's mostly constant, treat as Discharge.
        full_comment = " ".join([s['comment'] for s in segment_list])
        is_random_walk = 'random' in full_comment or 'walk' in full_comment or 'pulsed' in full_comment
        
        final_type = 'pulsed' if is_random_walk else 'discharge'
        
        # Stitch Time (Make it continuous)
        dfs = []
        cumulative_time = 0.0
        
        for item in segment_list:
            df = item['data'].copy()
            
            # Shift this step's time by the cumulative time of previous steps
            # relativeTime usually starts at 0 for each step
            df['Time'] = df['Time'] + cumulative_time
            
            # Update cumulative time for the next step
            if not df.empty:
                cumulative_time = df['Time'].iloc[-1] + (df['Time'].diff().mean() or 0.1)
            
            dfs.append(df)
            
        full_df = pd.concat(dfs, ignore_index=True)
        
        # Standardize columns
        # Force type='discharge' here so the standardizer calculates SoC correctly 
        # (Current > 0 = Discharge)
        df_std = DataStandardizer.standardize(full_df, config, 'discharge')
        
        desc = "Random Walk (Stitched)" if is_random_walk else "Reference Discharge (Stitched)"
        
        return {
            'type': final_type,
            'date': None,
            'data': df_std,
            'description': desc
        }
        
    def inspect_cycles(self, cycles: dict):
        """cycle inspection with validation info."""
        print(f"\n{'='*120}")
        print(f"{'CYCLE INSPECTION':<120}")
        print(f"{'='*120}")
        print(f"{'ID':<6} {'Type':<15} {'Duration':<12} {'Points':<8} {'Current':<15} {'SoC Range':<15} {'Description':<35}")
        print("-" * 120)
        
        for cid in sorted(cycles.keys()):
            c = cycles[cid]
            df = c['data']
            
            # Calculate stats
            duration_s = df['Time_s'].max() - df['Time_s'].min()
            duration_str = f"{duration_s/60:.1f}min"
            points = len(df)
            current_mean = df['Current_A'].mean()
            current_max = df['Current_A'].max()
            current_str = f"{current_mean:.2f}A(avg)"
            soc_start = df['SoC_measured'].iloc[0]
            soc_end = df['SoC_measured'].iloc[-1]
            soc_str = f"{soc_start:.2f}→{soc_end:.2f}"
            desc = c.get('description', '')[:33]
            
            # Add indicators for data quality
            if df['Current_A'].std() < 0.01:
                desc += " ⚠️  No pulse discharge detected!"
            
            print(f"{cid:<6} {c['type']:<15} {duration_str:<12} {points:<8} {current_str:<15} {soc_str:<15} {desc:<35}")
        
        print(f"{'='*120}\n")
        
        # Summary statistics
        total_cycles = len(cycles)
        types = {}
        for c in cycles.values():
            ctype = c['type']
            types[ctype] = types.get(ctype, 0) + 1
        
        print(f"Summary: {total_cycles} cycles total")
        for ctype, count in sorted(types.items()):
            print(f"  - {ctype}: {count} cycles")
        print()


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
            full_df = pd.concat(dfs, ignore_index=True)
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
        
        min_len = min(len(df_real), len(df_sim))
        df_real_plot = df_real.iloc[:min_len]
        df_sim_plot = df_sim.iloc[:min_len]
        
        ax1.plot(df_real_plot['Time_s'], df_real_plot['Voltage_V'], color='gray', label='Real')
        ax1.plot(df_sim_plot['Time_s'], df_sim_plot['V_model'], color='red', linestyle='--', label='Model')
        ax1.set_title(f"{title}\nRMSE={params.get('RMSE', 0):.4f}")
        ax1.legend(); ax1.grid(True)
        
        err = df_real_plot['Voltage_V'].values - df_sim_plot['V_model'].values
        ax2.plot(df_sim_plot['Time_s'], err, label='Error')
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
        
        if 'y1' in df_sim.columns and 'y2' in df_sim.columns:
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

class LoadProfileGenerator:
    @staticmethod
    def constant_current(current, duration, dt=0.1):
        t = np.arange(0, duration + dt, dt)
        return pd.DataFrame({'Time_s': t, 'Current_A': abs(current), 'dt': dt})

    @staticmethod
    def pulse_discharge(high_A, low_A, t_high, t_low, total_dur, dt=0.1):
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
        print("=" * 80)
        
        for i, cid in enumerate(dis_keys):
            df_cyc = cycles[cid]['data']
            opt = ModelOptimizer(model, df_cyc, config.nominal_capacity_Ah)
            
            if i > 0: 
                opt.anchor_params = current_params
                # Show what we're anchoring to
                print(f"\nAnchoring to previous cycle parameters")
            
            # Pass cycle number for adaptive anchoring
            best_p = opt.run_annealing(
                current_params, 
                initial_temp=1.0 if i==0 else 0.15,
                cycle_number=i
            )
            
            # status output
            rmse = best_p.get('_meta_loss', 0)
            acc = best_p.get('_meta_acc_rate', 0)
            
            # Calculate parameter changes
            if i > 0:
                changes = {k: ((best_p[k] - current_params[k]) / current_params[k] * 100) 
                        for k in ['Q', 'k', 'c', 'R_s', 'R1', 'C1']}
                
                print(f"\n Cycle {cid} ({i+1}/{len(dis_keys)})")
                print(f"   RMSE: {rmse:.8f} | Acceptance: {acc*100:.1f}%")
                print(f"   Q: {best_p['Q']:.8f} Ah ({changes['Q']:+.1f}%)")
                print(f"   R_s: {best_p['R_s']:.8f} Ω ({changes['R_s']:+.1f}%)")
                print(f"   c: {best_p['c']:.8f} ({changes['c']:+.1f}%)")
                print(f"   k: {best_p['k']:.8f} ({changes['k']:+.1f}%)")
                print(f"   R1: {best_p['R1']:.8f} Ω ({changes['R1']:+.1f}%)")
                print(f"   C1: {best_p['C1']:.8f} F ({changes['C1']:+.1f}%)")
                print(f"   tau: {best_p['R1'] * best_p['C1']:.8f}")
                
                # Flag large changes
                if abs(changes['c']) > 5:
                    print(f"⚠️  WARNING: Large change in 'c' parameter!")
                if abs(changes['Q']) > 5:
                    print(f"⚠️  WARNING: Large change in 'Q' parameter!")
                if abs(changes['R_s']) > 5:
                    print(f"⚠️  WARNING: Large change in 'R_s' parameter!")
                if abs(changes['k']) > 5:
                    print(f"⚠️  WARNING: Large change in 'k' parameter!")
                if abs(changes['R1']) > 5:
                    print(f"⚠️  WARNING: Large change in 'R1' parameter!")
                if abs(changes['C1']) > 5:
                    print(f"⚠️  WARNING: Large change in 'C1' parameter!")
            else:
                print(f"\n Cycle {cid} (Initial Fit)")
                print(f"   RMSE: {rmse:.4f} | Acceptance: {acc*100:.1f}%")
                print(f"   Q: {best_p['Q']:.3f} Ah")
            
            res = best_p.copy()
            res.update({'Cycle': cid, 'RMSE': rmse})
            results.append(res)
            current_params = best_p.copy()

        df_res = pd.DataFrame(results)
        path = self.store._safe_path(bid, "BatteryData", "parameter_evolution.csv")
        df_res.to_csv(path, index=False)
        print(f"\n Saved to {path}")
        
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
        dur = simTime
        calcTau = params['R1'] * params['C1']
        
        if choice == '1':
            amp = float(input("Amps: "))
            print(f"Using dt = {calcTau/userTauDivision}")
            df_prof = LoadProfileGenerator.constant_current(amp, dur, dt=calcTau/userTauDivision)
        else:
            h_A = float(input("High Amps: "))
            h_T = float(input("High Time: "))
            l_A = float(input("Low Amps (0=Rest): "))
            l_T = float(input("Low Time: "))
            print(f"Using dt = {calcTau/userTauDivision}")
            df_prof = LoadProfileGenerator.pulse_discharge(h_A, l_A, h_T, l_T, dur, dt=calcTau/userTauDivision)

        model = BatteryModel(df_ocv)
        df_sim = model.run(df_prof, params, cutoff_voltage=config.min_voltage_V)
        print(f"Runtime: {df_sim['Time_s'].max()/60:.1f} min")
        Visualizer.simulation_result(df_sim, config.min_voltage_V)
        
        # Save simulation results to file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filename = f"simulation_results_{timestamp}.csv"
        results_path = self.store._safe_path(bid, "BatteryData", results_filename)
        df_sim.to_csv(results_path, index=False)
        print(f"\n✓ Simulation results saved to: {results_filename}")

    def extract_ocv_pulsed(self, bid):
        """New method to handle pulsed OCV extraction specifically."""
        df_aging, df_imp, cycles = self.store.load_cycle_data(bid)
        config = self.store.load_config(bid)
        if not cycles: return

        # Find the pulsed cycle
        self.ingestor.inspect_cycles(cycles)
        
        try:
            cid = int(input("Enter Cycle ID of Pulsed Discharge (see list above): "))
            if cid not in cycles:
                print("Invalid Cycle ID.")
                return
            
            df_pulse = cycles[cid]['data']
            
            print("Analyzing pulses...")
            df_ocv = self.ocv_tools.create_ocv_from_pulse(df_pulse, config)
            
            plt.figure(figsize=(10, 6))
            plt.plot(df_ocv['SoC'], df_ocv['Voltage_OCV'], 'b.-', label='Extracted OCV')
            plt.title(f"Pulsed OCV Extraction (Cycle {cid})")
            plt.xlabel("State of Charge (SoC)")
            plt.ylabel("Voltage (V)")
            plt.grid(True)
            plt.legend()
            plt.show()
            
            if input("Save this OCV curve? [y/N]: ").lower() == 'y':
                path = self.store._safe_path(bid, "BatteryData", "ocv_reference.csv")
                df_ocv.to_csv(path, index=False)
                print("Saved.")
                
        except ValueError:
            print("Invalid input.")

class OCVTools:
    def extract_ocv(self, cycles, df_imp, config):
        dis_keys = [k for k,v in cycles.items() if v['type'] in ['discharge', 'low_current']]
        if not dis_keys: return None
        
        c_id = sorted(dis_keys, key=int)[0]
        print(f"Extracting OCV from Cycle {c_id} ({cycles[c_id]['type']})...")
        
        df = cycles[c_id]['data'].copy()
        
        r_est = 0.0
        if not df_imp.empty and 'Re' in df_imp.columns:
            r_est = df_imp.iloc[0]['Re'] + df_imp.iloc[0].get('Rct', 0)
        
        total_ah = (df['Current_A'] * df['dt']).sum() / 3600.0
        cum_ah = (df['Current_A'] * df['dt']).cumsum() / 3600.0
        df['SoC'] = 1.0 - (cum_ah / total_ah)
        
        df['Voltage_OCV'] = df['Voltage_V'] + (df['Current_A'] * r_est)
        return df[['SoC', 'Voltage_V', 'Voltage_OCV']].sort_values('SoC')
    
    def create_ocv_from_pulse(self, df_pulse: pd.DataFrame, config: BatteryConfig, 
                              min_rest_time: float = 20, current_thresh: float = 0.02):
        """
        Generates an OCV curve from a pulsed discharge cycle.
        
        Args:
            df_pulse: DataFrame containing the pulsed discharge data.
            min_rest_time: Minimum time (seconds) to consider a period as a 'rest'.
            current_thresh: Current (Amps) below which is considered zero/rest.
        """
        # Calculate Coulomb Counting for SoC
        total_capacity = (df_pulse['Current_A'] * df_pulse['dt']).sum() / 3600.0
        # If total capacity seems wrong (e.g. partial cycle), use nominal
        if total_capacity < config.nominal_capacity_Ah * 0.1:
            total_capacity = config.nominal_capacity_Ah
            
        cum_ah = (df_pulse['Current_A'] * df_pulse['dt']).cumsum() / 3600.0
        df_pulse['SoC_Calculated'] = 1.0 - (cum_ah / total_capacity)
        
        # Identify Rest Periods
        # Create a mask where current is approximately zero
        is_rest = df_pulse['Current_A'].abs() < current_thresh
        
        # Group consecutive rest periods
        # Perform a 'diff' on the boolean mask to find state changes
        rest_groups = is_rest.ne(is_rest.shift()).cumsum()
        
        ocv_points = []
        
        for grp_id, frame in df_pulse[is_rest].groupby(rest_groups):
            duration = frame['Time_s'].max() - frame['Time_s'].min()
            
            # Valid OCV point if rest is long enough
            if duration > min_rest_time:
                # Take the LAST point of the rest period (most relaxed voltage)
                last_point = frame.iloc[-1]
                ocv_points.append({
                    'SoC': last_point['SoC_Calculated'],
                    'Voltage_OCV': last_point['Voltage_V']
                })
        
        if not ocv_points:
            print(f"Warning: No rest periods detected (Current never dropped below {current_thresh}A).")
            # Return a generic fallback to prevent crash
            return pd.DataFrame({
                'SoC': [0.0, 1.0], 
                'Voltage_OCV': [config.min_voltage_V, config.max_voltage_V]
            })
        
        # Create DataFrame and Interpolate
        df_ocv = pd.DataFrame(ocv_points).sort_values('SoC')
        
        # Add 0% and 100% anchors if missing
        if df_ocv['SoC'].min() > 0.05:
            df_ocv = pd.concat([pd.DataFrame({'SoC': [0.0], 'Voltage_OCV': [config.min_voltage_V]}), df_ocv])
        if df_ocv['SoC'].max() < 0.95:
            df_ocv = pd.concat([df_ocv, pd.DataFrame({'SoC': [1.0], 'Voltage_OCV': [config.max_voltage_V]})])
            
        # Interpolate to create a smooth lookup table (e.g., 100 points)
        soc_interp = np.linspace(0, 1, 100)
        voc_interp = np.interp(soc_interp, df_ocv['SoC'], df_ocv['Voltage_OCV'])
        
        return pd.DataFrame({'SoC': soc_interp, 'Voltage_OCV': voc_interp})

class OCVToolsEnhanced(OCVTools):
    
    def extract_ocv_from_pulse(self, cycles: dict, config: BatteryConfig, 
                               pulse_cycle_id: int = None, rest_duration_min: float = 60.0):
        """
        Extract OCV curve from pulsed discharge data.
        
        The pulsed discharge format alternates between:
        1. Discharge pulse (D)
        2. Rest period (R)
        
        During rest, voltage relaxes toward OCV. This method extracts
        the relaxed voltage at the end of each rest period.
        
        Parameters:
        -----------
        cycles : dict
            Dictionary of cycle data
        config : BatteryConfig
            Battery configuration
        pulse_cycle_id : int, optional
            Specific pulsed cycle to use. If None, uses first pulsed cycle.
        rest_duration_min : float
            Minimum rest duration (seconds) to consider for OCV extraction
            
        Returns:
        --------
        pd.DataFrame with columns: SoC, Voltage_V, Voltage_OCV
        """
        
        # Find pulsed cycles
        pulsed_ids = [cid for cid, data in cycles.items() if data['type'] == 'pulsed']
        
        if not pulsed_ids:
            print("No pulsed discharge cycles found")
            return None
        
        if pulse_cycle_id is not None:
            if pulse_cycle_id not in pulsed_ids:
                print(f"Cycle {pulse_cycle_id} is not a pulsed cycle")
                return None
            cycle_id = pulse_cycle_id
        else:
            cycle_id = pulsed_ids[0]
        
        print(f"Extracting OCV from pulsed cycle {cycle_id}...")
        
        # Get the pulsed cycle data
        df = cycles[cycle_id]['data'].copy()
        
        # Detect discharge and rest segments
        # Rest: current near zero
        # Discharge: current > threshold
        current_threshold = 0.01  # Amps
        
        df['is_rest'] = df['Current_A'].abs() < current_threshold
        df['is_discharge'] = df['Current_A'] > current_threshold
        
        # Find rest periods
        # A rest period is a continuous sequence of is_rest=True
        df['rest_segment'] = (df['is_rest'] != df['is_rest'].shift()).cumsum()
        
        ocv_points = []
        
        # For each rest segment, extract the final voltage
        for seg_id in df[df['is_rest']]['rest_segment'].unique():
            seg_data = df[df['rest_segment'] == seg_id]
            
            # Check if this is actually a rest (not just noise)
            seg_duration = seg_data['Time_s'].max() - seg_data['Time_s'].min()
            
            if seg_duration < rest_duration_min:
                continue  # Too short to be meaningful rest
            
            # Get the last 10% of the rest period for stable OCV
            stable_portion = seg_data.tail(max(1, len(seg_data) // 10))
            
            # Average voltage during stable portion
            v_ocv = stable_portion['Voltage_V'].mean()
            
            # Get SoC at this point (from previous discharge)
            soc = seg_data['SoC_measured'].iloc[0]
            
            ocv_points.append({
                'SoC': soc,
                'Voltage_OCV': v_ocv,
                'Voltage_V': v_ocv,  # For compatibility
                'Time_s': seg_data['Time_s'].iloc[-1]
            })
        
        if not ocv_points:
            print("No valid OCV points extracted")
            return None
        
        df_ocv = pd.DataFrame(ocv_points)
        df_ocv = df_ocv.sort_values('SoC').reset_index(drop=True)
        
        # Remove duplicates and outliers
        df_ocv = df_ocv.drop_duplicates(subset=['SoC'])
        
        # Filter to valid SoC range
        df_ocv = df_ocv[(df_ocv['SoC'] >= 0) & (df_ocv['SoC'] <= 1)]
        
        print(f"Extracted {len(df_ocv)} OCV points from pulsed data")
        
        return df_ocv[['SoC', 'Voltage_V', 'Voltage_OCV']]
    
    def extract_ocv_from_reference(self, cycles: dict, df_imp: pd.DataFrame, config: BatteryConfig):
        """
        Extract OCV from reference discharge cycle (for RW or standard data).
        """
        return super().extract_ocv(cycles, df_imp, config)



# UI TOOLKIT FOR CLI


class UIHelper:
    """Helper class for user interface"""
    
    # Color codes (with fallback for terminals that don't support colors)
    COLORS = {
        'HEADER': '\033[95m',
        'BLUE': '\033[94m',
        'CYAN': '\033[96m',
        'GREEN': '\033[92m',
        'YELLOW': '\033[93m',
        'RED': '\033[91m',
        'BOLD': '\033[1m',
        'UNDERLINE': '\033[4m',
        'END': '\033[0m'
    }
    
    @staticmethod
    def colored(text, color='END'):
        """Add color to text if terminal supports it"""
        try:
            return f"{UIHelper.COLORS.get(color, '')}{text}{UIHelper.COLORS['END']}"
        except:
            return text
    
    @staticmethod
    def print_header(text, width=80):
        """Print a nice header"""
        print("\n" + "=" * width)
        print(text.center(width))
        print("=" * width)
    
    @staticmethod
    def print_success(text):
        """Print success message"""
        print(UIHelper.colored(f"✓ {text}", 'GREEN'))
    
    @staticmethod
    def print_warning(text):
        """Print warning message"""
        print(UIHelper.colored(f"⚠  {text}", 'YELLOW'))
    
    @staticmethod
    def print_error(text):
        """Print error message"""
        print(UIHelper.colored(f"✗ {text}", 'RED'))
    
    @staticmethod
    def print_info(text):
        """Print info message"""
        print(UIHelper.colored(f"ℹ  {text}", 'CYAN'))

def _edit_specific_parameter(key):
    """Sub-menu to edit a specific parameter's value, bounds, weight, and perturbation."""
    while True:
        # Fetch current state
        curr_val = SimulationDefaults.INITIAL_PARAMS.get(key)
        curr_bounds = SimulationDefaults.PARAM_BOUNDS.get(key, (0, 0))
        curr_weight = SimulationDefaults.ANCHOR_WEIGHTS.get(key, 0)
        curr_scale = SimulationDefaults.PERTURBATION_SCALES.get(key, 0.01)
        
        UIHelper.print_header(f"EDITING PARAMETER: {key}", 60)
        
        print(f"  [1] Initial Value    : {UIHelper.colored(str(curr_val), 'GREEN')}")
        print(f"  [2] Lower Bound      : {curr_bounds[0]}")
        print(f"  [3] Upper Bound      : {curr_bounds[1]}")
        print(f"  [4] Anchor Weight    : {curr_weight}")
        # Display as percentage for easier reading (0.01 -> 1.0%)
        print(f"  [5] Perturbation     : {curr_scale:.4f} ({curr_scale*100:.1f}%)")
        print(f"  [B] Back to Parameter List")
        
        sel = input("\nSelect option: ").strip().upper()
        
        if sel == 'B':
            break
            
        elif sel == '1': # Edit Value
            try:
                val = float(input(f"Enter new initial value for {key}: "))
                if not (curr_bounds[0] <= val <= curr_bounds[1]):
                    print(UIHelper.colored("Warning: New value is outside current bounds!", 'YELLOW'))
                SimulationDefaults.INITIAL_PARAMS[key] = val
                print("✓ Value updated.")
            except ValueError: print("Invalid number.")
            
        elif sel == '2': # Edit Min
            try:
                val = float(input(f"Enter new LOWER bound for {key}: "))
                if val >= curr_bounds[1]:
                    print(UIHelper.colored("Error: Lower bound must be less than upper bound.", 'RED'))
                else:
                    SimulationDefaults.PARAM_BOUNDS[key] = (val, curr_bounds[1])
                    print("✓ Lower bound updated.")
            except ValueError: print("Invalid number.")

        elif sel == '3': # Edit Max
            try:
                val = float(input(f"Enter new UPPER bound for {key}: "))
                if val <= curr_bounds[0]:
                    print(UIHelper.colored("Error: Upper bound must be greater than lower bound.", 'RED'))
                else:
                    SimulationDefaults.PARAM_BOUNDS[key] = (curr_bounds[0], val)
                    print("✓ Upper bound updated.")
            except ValueError: print("Invalid number.")

        elif sel == '4': # Edit Weight
            try:
                print("\n(Higher weight = Optimizer stays closer to initial value)")
                print("(Lower weight = Optimizer is free to explore more)")
                val = float(input(f"Enter new anchor weight for {key}: "))
                SimulationDefaults.ANCHOR_WEIGHTS[key] = val
                print("✓ Weight updated.")
            except ValueError: print("Invalid number.")

        elif sel == '5': # Edit Perturbation
            try:
                print("\n(Controls step size of random walk. 0.01 = 1% variation per step)")
                val = float(input(f"Enter new perturbation scale for {key} (e.g. 0.01): "))
                if val <= 0:
                     print(UIHelper.colored("Error: Scale must be positive.", 'RED'))
                else:
                    SimulationDefaults.PERTURBATION_SCALES[key] = val
                    print("✓ Perturbation scale updated.")
            except ValueError: print("Invalid number.")

def configure_physics_defaults():
    """Interactive menu to modify SimulationDefaults at runtime."""
    while True:
        UIHelper.print_header("PHYSICS PARAMETER CONFIGURATION", 100)
        
        # Expanded table header
        print(f"{'No.':<4} {'Param':<6} {'Value':<10} {'Bounds (Min, Max)':<20} {'Weight':<8} {'Perturb':<8}")
        print("-" * 75)
        
        # Display current values in a table
        params = list(SimulationDefaults.INITIAL_PARAMS.keys())
        for i, key in enumerate(params):
            val = SimulationDefaults.INITIAL_PARAMS.get(key)
            bounds = SimulationDefaults.PARAM_BOUNDS.get(key, "N/A")
            weight = SimulationDefaults.ANCHOR_WEIGHTS.get(key, 0)
            scale = SimulationDefaults.PERTURBATION_SCALES.get(key, 0.01)
            
            # Formatting for nice alignment
            b_str = f"({bounds[0]:.4f}, {bounds[1]:.4f})" if isinstance(bounds, tuple) else str(bounds)
            
            # Print row with new Perturb column
            print(f"[{i+1}]  {key:<6} {str(val):<10} {b_str:<20} {str(weight):<8} {scale:<8}")
            
        print("\n  [A] Algorithm Settings (Annealing Steps)")
        print("  [B] Back to Main Menu")
        
        choice = input("\nSelect parameter number to edit details, or option: ").strip().upper()
        
        if choice == 'B':
            break
            
        elif choice == 'A':
            # EDIT ANNEALING SETTINGS
            curr_steps = getattr(SimulationDefaults, 'ANNEALING_STEPS', 200)
            try:
                new_steps = int(input(f"Enter steps per temperature (Current: {curr_steps}): "))
                SimulationDefaults.ANNEALING_STEPS = new_steps
                print(f"Updated annealing steps to {new_steps}")
            except ValueError:
                print("Invalid number.")
                
        elif choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(params):
                _edit_specific_parameter(params[idx])
            else:
                print("Invalid selection.")

def show_help_menu():
    """Display comprehensive help information"""
    print("""
═════════════════════════════════════════════════════════════════════════════════
                    📚 MOBE BATTERY TOOLKIT - HELP                      
═════════════════════════════════════════════════════════════════════════════════


╔═══════════════════════════════════════════════════════════════════════════════╗
                            1. SETUP & DATA MANAGEMENT                         
╚═══════════════════════════════════════════════════════════════════════════════╝

┌─ Option 1: Create Battery / Import NASA Data ─────────────────────────────────┐
│                                                                               │
│  PURPOSE:                                                                     │
│    Initialize a new battery profile and optionally import experimental data   │
│    from NASA PCoE (Prognostics Center of Excellence) battery datasets.        │
│                                                                               │
│  WHAT IT DOES:                                                                │
│    • Creates a battery configuration with essential parameters:               │
│      - Battery ID: Unique identifier for this battery in the system           │
│      - Nominal Capacity (Ah): The rated capacity of the battery               │
│      - Max Voltage (V): Upper voltage limit (typically 4.2V for Li-ion)       │
│      - Min Voltage (V): Cutoff voltage below which battery is considered      │
│        depleted (typically 2.7V for Li-ion)                                   │
│       ( Max_V and Min_V depend on dataset and experiment )                    │
│                                                                               │
│    • Processes NASA .mat files to extract:                                    │
│      - Cycle-by-cycle discharge/charge data with time-series measurements     │
│      - Aging summary showing capacity fade over battery lifetime              │
│      - Impedance growth data (if available in the dataset)                    │
│      - Current, voltage, temperature, and time-stamped measurements           │
│                                                                               │
│  SUPPORTED DATASETS:                                                          │
│    The program has been tested with:                                          │
│       Dataset #5 (Batteries): B0005.mat, B0006.mat                            │
│        - Standard charge/discharge cycling at room temperature                │
│                                                                               │
│       Dataset #11 (Randomized Battery Usage - RW): RW3.mat                    │
│        - Usage patterns with variable loads                                   │
│        - Contains pulsed discharge sequences                                  │
│        - Includes low-current discharge cycles for OCV extraction             │
│                                                                               │
│    Other datasets from these categories should work, but may require manual   │
│    verification of cycle extraction quality.                                  │
│                                                                               │
│  DATA STRUCTURE CREATED:                                                      │
│    After import, the following files are generated in BatteryData/:           │
│      • battery_details.json - Configuration and metadata                      │
│      • aging_summary.csv - Capacity fade over cycles (if available)           │
│      • cycle_data_all.csv.gz - Compressed time-series data for all cycles     │
│      • impedance_summary.csv - EIS data if present in source                  │
│                                                                               │
│  WHERE TO GET NASA DATASETS:                                                  │
│       Official NASA Repository:                                               │
│       https://www.nasa.gov/intelligent-systems-division/                      │
│       discovery-and-systems-health/pcoe/pcoe-data-set-repository/             │
│                                                                               │
│       PHM Society Mirror:                                                     │
│       https://data.phmsociety.org/nasa/                                       │
│                                                                               │
│       Google Drive Archive:                                                   │
│       https://drive.google.com/drive/u/1/folders/                             │
│       1HtnDBGhMoAe53hl3t1XeCO9Z8IKd3-Q-                                       │
│                                                                               │
│  CREATING CUSTOM BATTERIES:                                                   │
│    If you want to create mock batteries or use custom data sources:           │
│      1. First import a NASA dataset to see the expected data structure        │
│      2. Examine the generated CSV files in BatteryData/ directory             │
│      3. Create your own files following the same format:                      │
│         - Time_s: Timestamp in seconds                                        │
│         - Current_A: Current in Amperes (positive = discharge)                │
│         - Voltage_V: Terminal voltage in Volts                                │
│         - Temperature_C: Cell temperature (optional)                          │
│         - dt: Time delta between measurements (calculated automatically)      │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘


┌─ Option 2: Inspect Cycles ────────────────────────────────────────────────────┐
│                                                                               │
│  PURPOSE:                                                                     │
│    Review and validate the quality of extracted battery cycling data.         │
│                                                                               │
│  WHAT IT DISPLAYS:                                                            │
│    • Complete cycle inventory with type classification:                       │
│      - discharge: Standard constant-current discharge cycles                  │
│      - low_current: Slow discharge cycles (ideal for OCV extraction)          │
│      - reference: Reference performance tests at standardized conditions      │
│      - pulsed: High-current pulses with rest periods                          │
│      - charge: Charging cycles ( Not extracted in RW datasets )               │
│                                                                               │
│    • For each cycle, shows:                                                   │
│      - Duration in hours/minutes                                              │
│      - Number of data points collected                                        │
│      - Average current draw                                                   │
│      - SoC Range (max to min)                                                 │
│      - Cycle Warnings                                                         │
│                                                                               │
│  USE CASES:                                                                   │
│    Verify data import was successful                                          │
│    Identify the best cycles for OCV extraction (look for low_current)         │
│    Select cycle for single model training                                     │
│    Detect anomalous cycles                                                    │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘


┌─ Option 3: Battery Information ───────────────────────────────────────────────┐
│                                                                               │
│  PURPOSE:                                                                     │
│    Display comprehensive status and configuration details for a battery.      │
│                                                                               │
│  CONFIGURATION SECTION:                                                       │
│    • Battery ID and source type                                               │
│    • Nominal capacity in Ah                                                   │
│    • Operating voltage range (min/max)                                        │
│                                                                               │
│  DATA STATUS:                                                                 │
│    • Total number of cycles loaded                                            │
│    • Breakdown by cycle type (discharge, charge, pulsed, etc.)                │
│                                                                               │
│  ANALYSIS STATUS:                                                             │
│    • OCV curve status:                                                        │
│      - Extracted (with number of SoC points)                                  │
│      - Not yet extracted (reminder to run Option 4)                           │
│                                                                               │
│    • Model training status:                                                   │
│      - Training complete (shows number of cycles trained)                     │
│       -- Average RMSE across all trained cycles                               │
│       -- Number of trained Cycles                                             │
│      - Not trained (reminder to run Option 6 or 7)                            │
│                                                                               │
│    Before starting any analysis or simulation, check this information to      │
│    ensure all prerequisite steps are complete. The analysis status section    │
│    clearly indicates what needs to be done next.                              │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘



╔═══════════════════════════════════════════════════════════════════════════════╗
║                      2. OCV (OPEN CIRCUIT VOLTAGE) EXTRACTION                 ║
╚═══════════════════════════════════════════════════════════════════════════════╝

CRITICAL PREREQUISITE: OCV extraction MUST be completed before any model training.

┌─ Option 4: Extract OCV (Standard Method) ─────────────────────────────────────┐
│                                                                               │
│  EXTRACTION PROCESS:                                                          │
│    1. Identifies the best discharge cycle for OCV extraction:                 │
│       • Prefers "low_current" cycles (typically 0.04A for RW datasets)        │
│       • Falls back to earliest "discharge" or "reference" cycle if needed     │
│       • Low currents minimize voltage drop due to internal resistance         │
│                                                                               │
│    2. Estimates internal resistance (R_total):                                │
│       • Uses first impedance data (Re + Rct) if available from measurements   │
│       • If not available, uses a conservative default estimate                │
│                                                                               │
│    3. Calculates State of Charge via coulomb counting:                        │
│       • Integrates current over time: ∫I(t)dt                                 │
│       • Normalizes to total cycle capacity                                    │
│       • SoC = 1 - (cumulative_Ah / total_capacity_Ah)                         │
│                                                                               │
│    4. Compensates for IR drop to get true OCV:                                │
│       • V_OCV ≈ V_terminal + (I × R_total)                                    │
│       • This removes the instantaneous voltage drop due to current flow       │
│                                                                               │
│    5. Creates a smooth lookup table:                                          │
│       • Saves as ocv_reference.csv for future use                             │
│                                                                               │
│  VERIFICATION PLOT:                                                           │
│    After extraction, a graph displays:                                        │
│      • SoC (0% to 100%) on X-axis                                             │
│      • OCV (Volts) on Y-axis                                                  │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘


┌─ Option 5: Extract OCV from Pulsed Data ──────────────────────────────────────┐
│                                                                               │
│  PURPOSE:                                                                     │
│    Extract OCV from pulsed discharge cycles where the battery alternates      │
│    between high-current discharge bursts and rest periods.                    │
│                                                                               │
│                                                                               │
│  HOW PULSED OCV EXTRACTION WORKS:                                             │
│    1. PULSE DETECTION:                                                        │
│       • Identifies discharge segments (high current)                          │
│       • Identifies rest segments (near-zero current)                          │
│       • Segments are defined by current threshold (typically 0.01-0.02A)      │
│                                                                               │
│    2. REST PERIOD ANALYSIS:                                                   │
│       • Only rest periods longer than minimum duration are considered         │
│         (typically 20 seconds minimum, but can be updated in the program)     │
│                                                                               │
│    3. VOLTAGE RELAXATION:                                                     │
│       • Takes the last 10% of each rest period as "stable OCV"                │
│       • This is when voltage has relaxed closest to true equilibrium          │
│       • Averages voltage over this stable portion to reduce noise             │
│                                                                               │
│    4. SOC ASSOCIATION:                                                        │
│       • Uses coulomb counting from start of cycle                             │
│       • Associates each rest period's OCV with its corresponding SoC          │
│                                                                               │
│    5. CURVE CONSTRUCTION:                                                     │
│       • Collects all valid OCV points                                         │
│       • Sorts by SoC and removes duplicates                                   │
│       • Interpolates to create smooth 100-point lookup table                  │
│                                                                               │
│  ADVANTAGES OVER STANDARD METHOD:                                             │
│    • Can extract OCV from real-world usage data                               │
│    • Doesn't require dedicated low-current discharge cycles                   │
│                                                                               │
│  LIMITATIONS:                                                                 │
│    • Requires sufficient rest time between pulses for voltage to relax        │
│    • May have fewer data points than continuous low-current discharge         │
│    • Quality depends on pulse profile design (duration, amplitude, rest time) │
│                                                                               │
│  INTERACTIVE PROCESS:                                                         │
│    1. Program shows all available cycles with their types                     │
│    2. You select the pulsed discharge cycle ID to analyze                     │
│    3. Program extracts OCV points and displays them graphically               │
│    4. You review the curve quality and choose to save or discard              │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘



╔═══════════════════════════════════════════════════════════════════════════════╗
║                        3. MODEL TRAINING & PARAMETER FITTING                  ║
╚═══════════════════════════════════════════════════════════════════════════════╝

MODEL ARCHITECTURE: Kinetic Battery Model (KiBaM) with RC Circuit

The complete model consists of:
  • Dual-tank KiBaM for charge distribution and rate effects
  • Series resistance (R_s) for instantaneous voltage drop
  • RC circuit (R1, C1) for transient voltage dynamics

Total Parameters: 6
  1. Q     - Total battery capacity (Ah)
  2. k     - Diffusion rate between tanks (1/s)
  3. c     - Fraction of capacity in available tank (dimensionless, 0-1)
  4. R_s   - Series resistance (Ohms)
  5. R1    - RC resistance (Ohms)
  6. C1    - RC capacitance (Farads)

┌─ Option 7: Fit Single Cycle ──────────────────────────────────────────────────┐
│                                                                               │
│  PURPOSE:                                                                     │
│    Train model parameters on one discharge cycle to understand the fitting    │
│    process and validate model behavior before full lifecycle training.        │
│                                                                               │
│  WHEN TO USE:                                                                 │
│    • First time using the toolkit (learning/testing)                          │
│    • To validate OCV curve quality before full training                       │
│    • To inspect parameter sensitivity on a specific cycle                     │
│    • To debug issues with specific problematic cycles                         │
│    • For quick parameter estimation when time is limited                      │
│                                                                               │
│  TRAINING PROCESS:                                                            │
│    1. CYCLE SELECTION:                                                        │
│       • Program lists all available discharge cycles                          │
│       • You choose one cycle ID to use as training data                       │
│                                                                               │
│    2. OPTIMIZATION ALGORITHM:                                                 │
│       • Uses scipy.optimize.minimize with L-BFGS-B method                     │
│       • Bounded optimization keeps parameters physically realistic            │
│       • Loss function: Root Mean Square Error (RMSE) between model and data   │
│                                                                               │
│    3. PARAMETER INITIALIZATION:                                               │
│       • Starts from default values                                            │
│       • Q initialized to nominal capacity from battery config                 │
│       • Other parameters from SimulationDefaults.INITIAL_PARAMS               │
│                                                                               │
│    4. CONVERGENCE:                                                            │
│       • Optimization runs until:                                              │
│         - Maximum iterations reached                                          │
│                                                                               │
│  OUTPUT & VISUALIZATION:                                                      │
│    After training completes, displays:                                        │
│      • Optimized parameter values for all 6 parameters                        │
│      • Final RMSE in millivolts (mV)                                          │
│      • Comparison plot with four subplots:                                    │
│                                                                               │
│        [1] Voltage vs. Time:                                                  │
│            • Blue line: Measured voltage from experimental data               │
│            • Red dashed: Model prediction with optimized parameters           │
│            • Shows overall voltage trajectory match                           │
│                                                                               │
│        [2] Voltage Error vs. Time:                                            │
│            • Difference between model and measurement                         │
│            • Ideally centered near zero with minimal spread                   │
│                                                                               │
│        [3] State of Charge vs. Time:                                          │
│            • Model's internal SoC estimate                                    │
│            • Should decrease smoothly from 1.0 to near 0.0                    │
│                                                                               │
│        [4] RC Voltage vs. Time:                                               │
│            • Transient voltage component from RC circuit                      │
│            • Shows how model captures dynamic response to current changes     │
│                                                                               │
│  WHAT GETS SAVED:                                                             │
│    Nothing is automatically saved. This is an exploratory/validation tool.    │
│    For permanent storage, use Option 6 (Lifecycle Analysis) instead.          │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘


┌─ Option 6: Lifecycle Analysis (Train All Cycles) ─────────────────────────────┐
│                                                                               │
│  PURPOSE:                                                                     │
│    Systematically train the battery model on every discharge cycle to track   │
│    how battery parameters evolve as the battery ages.                         │
│                                                                               │
│  WHAT IT DOES:                                                                │
│    Sequentially processes all discharge cycles in chronological order:        │
│      • Cycles 1, 2, 3, ..., N                                                 │
│      • For each cycle, runs parameter optimization (same as Option 7)         │
│      • Stores optimized parameters and goodness-of-fit metrics                │
│      • Generates parameter evolution plots showing aging trends               │
│                                                                               │
│  PARAMETER EVOLUTION TRACKING:                                                │
│    Creates a comprehensive database (parameter_evolution.csv) containing:     │
│      • Cycle number                                                           │
│      • All 6 model parameters (Q, k, c, R_s, R1, C1)                          │
│      • RMSE for that cycle                                                    │
│                                                                               │
│  VISUALIZATION OUTPUT:                                                        │
│    Generates a comprehensive multi-panel plot:                                │
│      1. Q vs. Cycle: Capacity fade trajectory                                 │
│      2. k vs. Cycle: Diffusion rate evolution                                 │
│      3. c vs. Cycle: Available charge fraction changes                        │
│      4. R_s vs. Cycle: Resistance growth over time                            │
│      5. R1 vs. Cycle: RC resistance evolution                                 │
│      6. C1 vs. Cycle: Capacitance changes                                     │
│      7. RMSE vs. Cycle: Model accuracy over lifecycle                         │
│                                                                               │
│    Each subplot includes:                                                     │
│      • Scatter points for individual cycle values                             │
│      • Trend line showing overall aging trajectory                            │
│                                                                               │
│                                                                               │
│  WHAT GETS SAVED:                                                             │
│    File: BatteryData/<battery_id>/parameter_evolution.csv                     │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘



╔═══════════════════════════════════════════════════════════════════════════════╗
║                            4. ANALYSIS & VALIDATION                           ║
╚═══════════════════════════════════════════════════════════════════════════════╝

┌─ Option 9: Manual Simulation (Validation Mode) ───────────────────────────────┐
│                                                                               │
│  PURPOSE:                                                                     │
│    Validate model accuracy by manually specifying parameters and comparing    │
│    model predictions against actual experimental discharge data.              │
│                                                                               │
│  WORKFLOW:                                                                    │
│    1. SELECT VALIDATION CYCLE:                                                │
│       • Choose a discharge cycle from your dataset                            │
│       • This cycle provides the "ground truth" voltage profile                │
│                                                                               │
│    2. INPUT PARAMETERS:                                                       │
│         Manual Entry:                                                         │
│             • Directly type each of the 6 parameters                          │
│                                                                               │
│    3. MODEL EXECUTION:                                                        │
│       • Runs forward simulation using input parameters                        │
│       • Uses the actual current profile from the selected cycle               │
│       • Initial SoC estimated from first voltage measurement                  │
│       • Applies voltage cutoff when specified minimum is reached              │
│                                                                               │
│    4. COMPARISON & METRICS:                                                   │
│       Generates comparison showing:                                           │
│         • Voltage overlay plot (model vs. measured)                           │
│         • Error statistics:                                                   │
│           - RMSE (Root Mean Square Error) in mV                               │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘



╔═══════════════════════════════════════════════════════════════════════════════╗
║                       5. SIMULATION & PREDICTION                              ║
╚═══════════════════════════════════════════════════════════════════════════════╝

┌─ Option 10: Simulation Playground (Predictive Mode) ──────────────────────────┐
│                                                                               │
│  PURPOSE:                                                                     │
│    Simulate battery behavior under hypothetical load profiles to predict      │
│    runtime, voltage response, and end-of-discharge characteristics.           │
│                                                                               │
│  KEY DIFFERENCE FROM OPTION 9:                                                │
│    • Option 9: Validates model against KNOWN experimental data                │
│    • Option 10: Predicts behavior for UNKNOWN future scenarios                │
│                                                                               │
│  SIMULATION CONFIGURATION:                                                    │
│                                                                               │
│    STEP 1: Parameter Selection                                                │
│      [1] Manual Parameters:                                                   │
│          • Enter all 6 parameters manually                                    │
│          • Use this for "what-if" scenarios with hypothetical parameters      │
│                                                                               │
│      [2] Load from Lifecycle Training:                                        │
│          • Choose a cycle number from parameter_evolution.csv                 │
│          • Simulates battery at that specific age/health state                │
│          • Example: "How long would this battery last if I used it when it    │
│            was at Cycle 250?" → Select Cycle 250 parameters                   │
│                                                                               │
│    STEP 2: Load Profile Definition                                            │
│      [1] Constant Current Discharge:                                          │
│          • Single, steady current draw                                        │
│          • Input: Current in Amperes (e.g., 2.0A)                             │
│          • Simulates until voltage cutoff or max time reached                 │
│                                                                               │
│      [2] Pulse Discharge:                                                     │
│          • Alternating high and low current periods                           │
│          • Inputs:                                                            │
│            - High current (Amps): Peak discharge rate                         │
│            - High duration (seconds): How long each pulse lasts               │
│            - Low current (Amps): Base load (0 = complete rest)                │
│            - Low duration (seconds): Rest/low-load period                     │
│                                                                               │
│  TIME STEP CONTROL:                                                           │
│    Simulation uses adaptive time stepping based on RC time constant:          │
│      dt = τ / userTauDivision, where τ = R1 × C1                              │
│                                                                               │
│    Higher userTauDivision → smaller time steps → more accurate but slower     │
│    Default = 4                                                                │
│    Can be changed in Option 12 (Simulation Settings)                          │
│                                                                               │
│  SIMULATION EXECUTION:                                                        │
│    The model integrates the KiBaM equations forward in time:                  │
│      • Updates y1 (available charge) and y2 (bound charge)                    │
│      • Calculates diffusion between tanks                                     │
│      • Computes RC circuit voltage                                            │
│      • Looks up OCV from SoC                                                  │
│      • Calculates terminal voltage: V = V_OCV - I×R_s - V_RC                  │
│      • Stops when V < V_cutoff or max simulation time reached                 │
│                                                                               │
│  FILE SAVING:                                                                 │
│    After simulation, results are automatically saved to:                      │
│      BatteryData/<battery_id>/simulation_results_YYYYMMDD_HHMMSS.csv          │
│                                                                               │
│    This file contains complete time-series data:                              │
│      • Time_s: Timestamp                                                      │
│      • Current_A: Applied current                                             │
│      • V_model: Terminal voltage                                              │
│      • SoC_model: State of charge                                             │
│      • V_rc: RC voltage component                                             │
│      • y1: Available charge (Coulombs)                                        │
│      • y2: Bound charge (Coulombs)                                            │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘



╔═══════════════════════════════════════════════════════════════════════════════╗
║                           6. SYSTEM UTILITIES                                 ║
╚═══════════════════════════════════════════════════════════════════════════════╝

┌─ Option 11: Delete Battery ───────────────────────────────────────────────────┐
│                                                                               │
│  PURPOSE:                                                                     │
│    Permanently remove a battery and all associated data from the system.      │
│                                                                               │
│  WARNING: This action is IRREVERSIBLE!                                        │
│                                                                               │
│  WHAT GETS DELETED:                                                           │
│    Removes the entire battery directory, including:                           │
│      • battery_details.json (configuration)                                   │
│      • aging_summary.csv (capacity fade data)                                 │
│      • cycle_data_all.csv.gz (all experimental measurements)                  │
│      • impedance_summary.csv (EIS data)                                       │
│      • ocv_reference.csv (OCV curve)                                          │
│      • parameter_evolution.csv (training results)                             │
│      • simulation_results_*.csv (all saved simulations)                       │
│      • Any plots or temporary files                                           │
│                                                                               │
│  SAFETY FEATURES:                                                             │
│    • Requires explicit confirmation (type 'y' to confirm)                     │
│    • Does not proceed if anything other than 'y' is entered                   │
│    • Displays warning message in red before deletion                          │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘


┌─ Option 12: Configure Simulation Settings ────────────────────────────────────┐
│                                                                               │
│  PURPOSE:                                                                     │
│    Adjust global simulation parameters that affect accuracy and computation   │
│    time. These settings apply to all future simulations until changed again.  │
│                                                                               │
│     NOT SAVED BETWEEN DIFFERENT PROGRAM EXECUTIONS                            │
│                                                                               │
│  SETTING 1: TAU DIVISIONS (Time Step Control)                                 │
│                                                                               │
│    What it controls:                                                          │
│      The simulation time step is calculated as: dt = τ / userTauDivision      │
│      where τ = R1 × C1 is the RC time constant                                │
│                                                                               │
│    Effect on simulation:                                                      │
│      Higher value → Smaller time steps → More accuracy, slower computation    │
│      Lower value → Larger time steps → Faster computation, less accuracy      │
│                                                                               │
│  SETTING 2: MAXIMUM SIMULATION TIME                                           │
│                                                                               │
│    What it controls:                                                          │
│      Maximum simulation duration in hours before automatic termination.       │
│      Default: 30 hours (108,000 seconds)                                      │
│                                                                               │
│    Purpose:                                                                   │
│      • Prevents infinite loops if battery never reaches cutoff                │
│      • Limits computational cost for slow discharge scenarios                 │
│      • Acts as a safety timeout                                               │
│                                                                               │
│  NOTES:                                                                       │
│    • These settings are global and persist across all battery simulations     │
│    • They do NOT affect model training (Options 6, 7)                         │
│    • You can change them at any time; changes take effect immediately         │
│    • Default values are suitable for most common use cases                    │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘


                
┌─ Option 13: Configure Physics Defaults ───────────────────────────────────────┐
│                                                                               │
│  PURPOSE:                                                                     │
│    Advanced configuration of the physics engine and optimization algorithm.   │
│                                                                               │
│  SETTING 1: PHYSICS PARAMETER CONFIGURATION                                   │
│    Select any parameter (Q, k, c, R_s, R1, C1) to open its sub-menu:          │
│                                                                               │
│    A. Initial Value: Starting guess for the optimizer.                        │
│                                                                               │
│    B. Bounds (Min/Max): Hard constraints for the physics engine.              │
│                                                                               │
│    C. Anchor Weights: (Exploitation)                                          │
│       Controls how "sticky" a parameter is to the previous cycle.             │
│       High Weight = Parameter resists changing.                               │
│                                                                               │
│    D. Perturbation Scale: (Exploration)                                       │
│       Controls the "step size" of the random walk during optimization.        │
│       • Value is a fraction of the current parameter (e.g., 0.01 = 1%).       │
│       • High (e.g., 0.05): Large jumps. Good for finding global minima        │
│         but might be unstable.                                                │
│       • Low (e.g., 0.001): Fine tuning. Good for precision but might          │
│         get stuck in local minima.                                            │
│                                                                               │
│  SETTING 2: ANNEALING STEPS                                                   │
│      The number of iterations per temperature level.                          │
│                                                                               │
└───────────────────────────────────────────────────────────────────────────────┘

          

╔═══════════════════════════════════════════════════════════════════════════════╗
║                         7. TECHNICAL BACKGROUND                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝

KINETIC BATTERY MODEL (KiBaM) - FUNDAMENTAL CONCEPTS:
─────────────────────────────────────────────────────────────────────────────────

The KiBaM is an empirical model that captures:
  • Rate effects: Battery delivers less capacity at high currents
  • Recovery effects: Capacity "recovers" during rest or low-current periods
  • Non-ideal charge redistribution within the battery

Dual-Tank Analogy:
  ┌─────────────────────┐      ┌─────────────────────┐
  │  AVAILABLE TANK     │◄────►│   BOUND TANK        │
  │     (y1)            │  k   │     (y2)            │
  │   Fraction: c       │      │  Fraction: (1-c)    │
  └──────┬──────────────┘      └─────────────────────┘
         │ Discharge
         │ Current (I)
         ▼

  • y1: Charge immediately available for discharge
  • y2: Charge that must first diffuse to y1 before use
  • k: Diffusion rate between tanks (1/s)
  • c: Fraction of total capacity in available tank

Key Equations:
  dy1/dt = -I + k(h2 - h1)
  dy2/dt = -k(h2 - h1)
  where h1 = y1/c and h2 = y2/(1-c)

RC CIRCUIT - TRANSIENT VOLTAGE DYNAMICS:
─────────────────────────────────────────────────────────────────────────────────

Captures the fast voltage response to current changes:
  
  V_RC = R1 × I × (1 - e^(-t/τ))
  where τ = R1 × C1

  • R1: Charge transfer resistance (Ohms)
  • C1: Double-layer capacitance (Farads)
  • τ: Time constant (seconds) - how fast voltage responds

Physical interpretation:
  • R1 relates to electrochemical kinetics at electrode/electrolyte interface
  • C1 represents double-layer charging
  • Together, they model the transient response when load is applied

COMPLETE VOLTAGE MODEL:
─────────────────────────────────────────────────────────────────────────────────

  V_terminal = V_OCV(SoC) - I×R_s - V_RC

  Where:
    • V_OCV: Open circuit voltage (looked up from extracted curve)
    • SoC: (y1 + y2) / Q_total
    • R_s: Series resistance (instantaneous IR drop)
    • V_RC: Transient voltage from RC circuit

This equation shows three voltage components:
  1. Thermodynamic voltage (V_OCV): Depends only on SoC
  2. Ohmic drop (I×R_s): Proportional to current
  3. Transient drop (V_RC): Time-dependent response

PARAMETER PHYSICAL MEANING
─────────────────────────────────────────────────────────────────────────────────

Q (Capacity):
  • Units: Ah (Ampere-hours)
  • Typical: 1.5 - 3.0 Ah for 18650 cells
  • Should match nominal capacity from datasheet initially
  • Decreases monotonically with aging

k (Diffusion Rate):
  • Units: 1/s (per second)
  • Typical: 0.0001 - 0.01 (wide range)
  • Higher k → faster equilibration between tanks
  • Lower k → more pronounced rate effects
  • May decrease with aging due to electrode degradation

c (Available Fraction):
  • Dimensionless, range: 0.4 - 0.6
  • Typical: ~0.5 (50% in each tank)
  • Represents effective porosity / active material distribution
  • Relatively stable over battery life

R_s (Series Resistance):
  • Units: Ohms (Ω)
  • Typical: 0.05 - 0.3 Ω for Li-ion cells
  • Includes: contact resistance, current collector, electrolyte bulk
  • Increases with aging (SEI growth, loss of electrical connectivity)

R1 (RC Resistance):
  • Units: Ohms (Ω)
  • Typical: 0.05 - 0.4 Ω
  • Charge transfer resistance at electrode surface
  • Increases with aging (surface degradation, loss of active sites)

C1 (RC Capacitance):
  • Units: Farads (F)
  • Typical: 10 - 100 F
  • Double-layer capacitance
  • Changes reflect alterations in electrochemically active surface area


OPTIMIZATION ALGORITHM DETAILS:
─────────────────────────────────────────────────────────────────────────────────

Method: L-BFGS-B (Limited-memory Broyden-Fletcher-Goldfarb-Shanno with Bounds)

Loss Function:
  L = w_RMSE × RMSE(V_model, V_measured) + Σ w_i × (p_i - p_i_default)²
  
  Components:
    • RMSE term: Fit to voltage data
    • Anchor terms: Regularization to prevent unrealistic parameters

═══════════════════════════════════════════════════════════════════════════════════
                                    END OF HELP                                     
═══════════════════════════════════════════════════════════════════════════════════    
    """)
    
    input("\nPress Enter to return to main menu...")


def show_battery_info(battery_id, store):
    """Display detailed information about a battery"""
    UIHelper.print_header(f"BATTERY INFORMATION: {battery_id}", 90)
    
    config = store.load_config(battery_id)
    if not config:
        UIHelper.print_error(f"Battery {battery_id} not found!")
        return
    
    df_aging, df_imp, cycles = store.load_cycle_data(battery_id)
    
    print("\n" + UIHelper.colored("═══ CONFIGURATION ═══", 'BOLD'))
    print(f"  Battery ID: {UIHelper.colored(config.battery_id, 'CYAN')}")
    print(f"  Nominal Capacity: {UIHelper.colored(f'{config.nominal_capacity_Ah:.2f} Ah', 'GREEN')}")
    print(f"  Voltage Range: {config.min_voltage_V:.2f}V - {config.max_voltage_V:.2f}V")
    
    print("\n" + UIHelper.colored("═══ DATA STATUS ═══", 'BOLD'))
    if cycles and len(cycles) > 0:
        UIHelper.print_success(f"Cycle data loaded: {len(cycles)} cycles")
        type_counts = {}
        for cid, cdata in cycles.items():
            ctype = cdata.get('type', 'unknown')
            type_counts[ctype] = type_counts.get(ctype, 0) + 1
        
        print("\n  Cycle breakdown:")
        for ctype, count in sorted(type_counts.items()):
            print(f"    • {ctype}: {UIHelper.colored(str(count), 'CYAN')} cycles")
    else:
        UIHelper.print_warning("No cycle data loaded")
    
    print("\n" + UIHelper.colored("═══ ANALYSIS STATUS ═══", 'BOLD'))
    ocv_path = store._safe_path(battery_id, "BatteryData", "ocv_reference.csv")
    if os.path.exists(ocv_path):
        UIHelper.print_success("OCV curve extracted")
        df_ocv = pd.read_csv(ocv_path)
        print(f"    Points: {len(df_ocv)}")
    else:
        UIHelper.print_warning("OCV curve not yet extracted (run Option 4 or 5)")
    
    params_path = store._safe_path(battery_id, "BatteryData", "parameter_evolution.csv")
    if os.path.exists(params_path):
        UIHelper.print_success("Model training completed")
        df_params = pd.read_csv(params_path)
        print(f"    Cycles trained: {len(df_params)}")
        print(f"    Average RMSE: {df_params['RMSE'].mean()*1000:.1f} mV")
    else:
        UIHelper.print_warning("Model not yet trained (run Option 6 or 7)")
    
    input("\nPress Enter to continue...")


def inspect_cycles_ui(battery_id, store, sim_ctrl):
    """User interface for cycle inspection"""
    UIHelper.print_header(f"CYCLE INSPECTION: {battery_id}", 90)
    
    df_aging, df_imp, cycles = store.load_cycle_data(battery_id)
    
    if not cycles or len(cycles) == 0:
        UIHelper.print_error("No cycle data available!")
        return
    
    sim_ctrl.ingestor.inspect_cycles(cycles)
    input("\nPress Enter to continue...")


# MAIN UI

def main():
    store = StorageManager()
    sim_ctrl = SimulationController()
    
    while True:
        # Clear screen effect (optional)
        print("\n" * 2)
        
        # Header
        UIHelper.print_header("⚡ MOBE BATTERY TOOLKIT ⚡", 90)
        
        # Show available batteries
        batteries = store.get_list()
        if batteries:
            print(f"\n{UIHelper.colored('Available Batteries:', 'BOLD')} {', '.join([UIHelper.colored(b, 'CYAN') for b in batteries])}")
        else:
            print(f"\n{UIHelper.colored('No batteries loaded', 'YELLOW')} - Create one with Option 1")
        
        # Menu organized by category
        print(f"\n{UIHelper.colored('═══ SETUP & DATA ═══', 'HEADER')}")
        print(f"  {UIHelper.colored('1', 'CYAN')}. Create Battery / Import NASA Data")
        print(f"      └─ Set up new battery and load .mat file")
        print(f"  {UIHelper.colored('2', 'CYAN')}. Inspect Cycles")
        print(f"      └─ View all cycles with types, durations, quality")
        print(f"  {UIHelper.colored('3', 'CYAN')}. Battery Information")
        print(f"      └─ Show detailed info, analysis status, file locations")
        
        print(f"\n{UIHelper.colored('═══ OCV EXTRACTION ═══', 'HEADER')}")
        print(f"  {UIHelper.colored('4', 'CYAN')}. Extract OCV Curve")
        print(f"      └─ From low current discharge (best) or reference cycle")
        print(f"  {UIHelper.colored('5', 'CYAN')}. Extract OCV from Pulse")
        print(f"      └─ Alternative method using pulsed discharge rest periods")
        
        print(f"\n{UIHelper.colored('═══ MODEL TRAINING ═══', 'HEADER')}")
        print(f"  {UIHelper.colored('6', 'CYAN')}. Lifecycle Analysis (Train All)")
        print(f"      └─ Train on all cycles, track degradation (5-30 min)")
        print(f"  {UIHelper.colored('7', 'CYAN')}. Fit Single Cycle (Train One)")
        print(f"      └─ Quick training on one cycle for testing")
        
        print(f"\n{UIHelper.colored('═══ ANALYSIS & VISUALIZATION ═══', 'HEADER')}")
        print(f"  {UIHelper.colored('8', 'CYAN')}. Dashboard (Visualize Results)")
        print(f"      └─ Capacity fade, parameter evolution, cycle comparison")
        print(f"  {UIHelper.colored('9', 'CYAN')}. Manual Simulation (Test Params)")
        print(f"      └─ Test specific parameters, sensitivity analysis")
        
        print(f"\n{UIHelper.colored('═══ PREDICTION & SIMULATION ═══', 'HEADER')}")
        print(f"  {UIHelper.colored('10', 'CYAN')}. Simulation Playground (Predict)")
        print(f"      └─ Predict behavior under new current profiles")
        
        print(f"\n{UIHelper.colored('═══ SYSTEM ═══', 'HEADER')}")
        print(f"  {UIHelper.colored('11', 'CYAN')}. Delete Battery")
        print(f"  {UIHelper.colored('12', 'CYAN')}. Configure Simulation Settings")
        print(f"      └─ Tau divisions, max time, etc.")
        print(f"  {UIHelper.colored('13', 'CYAN')}. Configure Physics Defaults")
        print(f"  {UIHelper.colored('h', 'CYAN')}. Help & Documentation")
        print(f"      └─ Comprehensive guide to all features")
        print(f"  {UIHelper.colored('q', 'CYAN')}. Quit")
        
        print("\n" + "=" * 90)
        sel = input(f"{UIHelper.colored('Select option:', 'BOLD')} ").lower().strip()
        
        if sel == 'q':
            print(f"\n{UIHelper.colored('👋 Goodbye!', 'GREEN')}")
            break
        
        # Help menu
        if sel == 'h' or sel == 'help':
            show_help_menu()
            continue
        
        # Dashboard
        if sel == '8':
            bid = input("Battery ID: ").strip()
            if not bid:
                UIHelper.print_warning("Please enter a battery ID")
                continue
            if bid not in batteries:
                UIHelper.print_error(f"Battery '{bid}' not found!")
                continue
            
            UIHelper.print_info(f"Loading dashboard for {bid}...")
            df_a, df_i, cyc = store.load_cycle_data(bid)
            conf = store.load_config(bid)
            if conf:
                Visualizer.dashboard(bid, df_a, df_i, cyc, conf)
            else:
                UIHelper.print_error("Could not load configuration")
        
        # Create/Import Battery
        elif sel == '1':
            UIHelper.print_header("CREATE NEW BATTERY", 90)
            
            bid = input("New Battery ID: ").strip()
            if not bid:
                UIHelper.print_warning("Battery ID cannot be empty")
                continue
            
            if bid in batteries:
                UIHelper.print_warning(f"Battery '{bid}' already exists!")
                if input("Overwrite? [y/N]: ").lower() != 'y':
                    continue
            
            print("\n" + UIHelper.colored("--- Configuration ---", 'BOLD'))
            try:
                cap = float(input("Nominal Capacity (Ah) [2.0]: ") or 2.0)
                v_max = float(input("Max Voltage (V) [4.2]: ") or 4.2)
                v_min = float(input("Min Voltage (V) [2.7]: ") or 2.7)
            except ValueError:
                UIHelper.print_error("Invalid input. Please enter numbers.")
                continue
            
            conf = BatteryConfig(bid, cap, v_min, v_max)
            store.create_battery(conf)
            UIHelper.print_success(f"Created battery: {bid}")
            
            if input("\nImport NASA .mat file now? [y/N]: ").lower() == 'y':
                path = input("Full path to .mat file: ").strip('"').strip("'")
                
                if not os.path.exists(path):
                    UIHelper.print_error(f"File not found: {path}")
                    continue
                
                UIHelper.print_info("Processing .mat file...")
                df_a, df_i, cyc = sim_ctrl.ingestor.process_file(path, conf)
                
                if cyc and len(cyc) > 0:
                    store.save_cycle_data(bid, df_a, df_i, cyc)
                    UIHelper.print_success(f"Import successful: {len(cyc)} cycles loaded")
                    
                    # Show cycle summary
                    type_counts = {}
                    for cdata in cyc.values():
                        ctype = cdata.get('type', 'unknown')
                        type_counts[ctype] = type_counts.get(ctype, 0) + 1
                    
                    print("\n  Cycle breakdown:")
                    for ctype, count in sorted(type_counts.items()):
                        print(f"    • {ctype}: {count} cycles")
                else:
                    UIHelper.print_error("Import failed: No valid cycles found")
        
        # Extract OCV
        elif sel == '4':
            bid = input("Battery ID: ").strip()
            if bid not in batteries:
                UIHelper.print_error(f"Battery '{bid}' not found!")
                continue
            
            UIHelper.print_info(f"Extracting OCV curve for {bid}...")
            sim_ctrl.extract_ocv(bid)
        
        # Lifecycle Analysis
        elif sel == '6':
            bid = input("Battery ID: ").strip()
            if bid not in batteries:
                UIHelper.print_error(f"Battery '{bid}' not found!")
                continue
            
            UIHelper.print_info(f"Starting lifecycle analysis for {bid}...")
            UIHelper.print_warning("This may take a long time depending on cycle count")
            sim_ctrl.run_lifecycle(bid)
        
        # Fit Single Cycle
        elif sel == '7':
            bid = input("Battery ID: ").strip()
            if bid not in batteries:
                UIHelper.print_error(f"Battery '{bid}' not found!")
                continue
            
            sim_ctrl.fit_single_cycle(bid)
        
        # Manual Simulation
        elif sel == '9':
            bid = input("Battery ID: ").strip()
            if bid not in batteries:
                UIHelper.print_error(f"Battery '{bid}' not found!")
                continue
            
            sim_ctrl.run_manual_sim(bid)
        
        # Simulation Playground
        elif sel == '10':
            bid = input("Battery ID: ").strip()
            if bid not in batteries:
                UIHelper.print_error(f"Battery '{bid}' not found!")
                continue
            
            sim_ctrl.playground(bid)
        
        # Delete Battery
        elif sel == '11':
            bid = input("Battery ID to delete: ").strip()
            if bid not in batteries:
                UIHelper.print_error(f"Battery '{bid}' not found!")
                continue
            
            confirm = input(f"{UIHelper.colored('⚠ WARNING:', 'RED')} Delete '{bid}' permanently? [y/N]: ")
            if confirm.lower() == 'y':
                store.delete_battery(bid)
                UIHelper.print_success(f"Deleted battery: {bid}")
            else:
                UIHelper.print_info("Deletion cancelled")
        
        # Configure Simulation Settings
        elif sel == '12':
            UIHelper.print_header("SIMULATION SETTINGS", 90)
            print("\n1. Set Tau Divisions (time step control)")
            print("2. Set Maximum Simulation Time")
            print("3. Back to main menu")
            
            sub_sel = input("\nSelect: ").strip()
            
            if sub_sel == '1':
                try:
                    global userTauDivision
                    userinput = int(input("Tau divisions (1 to N, higher = finer) [4]: ") or 4)
                    userTauDivision = userinput
                    UIHelper.print_success(f"Tau divisions set to {userTauDivision}")
                except ValueError:
                    UIHelper.print_error("Invalid input. Please enter an integer.")
            
            elif sub_sel == '2':
                try:
                    global simTime
                    userinput = float(input("Max simulation time (hours) [30]: ") or 30)
                    simTime = userinput * 3600
                    UIHelper.print_success(f"Max simulation time set to {userinput:.1f} hours ({simTime:.0f} seconds)")
                except ValueError:
                    UIHelper.print_error("Invalid input. Please enter a number.")
        
        # Extract OCV from Pulse
        elif sel == '5':
            bid = input("Battery ID: ").strip()
            if bid not in batteries:
                UIHelper.print_error(f"Battery '{bid}' not found!")
                continue
            
            UIHelper.print_info(f"Extracting OCV from pulsed data for {bid}...")
            sim_ctrl.extract_ocv_pulsed(bid)
        
        # Inspect Cycles
        elif sel == '2':
            bid = input("Battery ID: ").strip()
            if bid not in batteries:
                UIHelper.print_error(f"Battery '{bid}' not found!")
                continue
            
            inspect_cycles_ui(bid, store, sim_ctrl)
        
        # Battery Information
        elif sel == '3':
            bid = input("Battery ID: ").strip()
            if bid not in batteries:
                UIHelper.print_error(f"Battery '{bid}' not found!")
                continue
            
            show_battery_info(bid, store)

        elif sel == '13':
             configure_physics_defaults()
        
        else:
            UIHelper.print_warning(f"Invalid option: '{sel}'. Type 'h' for help.")


if __name__ == "__main__":
    main()