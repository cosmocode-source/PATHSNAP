"""Run the largest training trip and largest known network through the API."""
import ctypes
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'backend'), str(ROOT)]
from app.config import settings
from app.main import app
from app.model_loader import FINAL_HMM_MODULE


def peak_working_set_mb():
    """Windows peak process working set; includes imports and earlier work."""
    if os.name != 'nt':
        return None
    class Counters(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong),
                    ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                    ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                    ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                    ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]
    memory = Counters(); memory.cb = ctypes.sizeof(memory)
    ctypes.windll.kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    ctypes.windll.psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    ok = ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(memory), memory.cb)
    return round(memory.PeakWorkingSetSize / (1024*1024), 2) if ok else None


def test_largest_training_trip_and_network():
    settings.MODEL_DIR = ROOT / 'trained_models/final_model'
    settings.HMM_MODULE = FINAL_HMM_MODULE
    settings.HMM_CONFIG = ROOT / 'trained_models/hmm/config.json'
    settings.STRICT_HMM = True
    settings.TRIPS_DIR = ROOT / 'Dataset/train'
    inventory = pd.read_csv(ROOT / 'data_preprocessing/cleaned_data/validation/trip_inventory.csv', dtype={'trip_id':str})
    inventory = inventory[inventory['split'] == 'train']
    largest_trip = inventory.loc[inventory.n_gps.idxmax(), 'trip_id']
    largest_network = inventory.loc[inventory.n_arcs.idxmax(), 'trip_id']
    records = []
    with TestClient(app) as client:
        for kind, trip in [('largest_network',largest_network), ('largest_train_trip',largest_trip)]:
            response = client.post('/api/predict', json={'trip_id':trip, 'refresh':True})
            assert response.status_code == 202
            job_id = response.json()['job_id']
            deadline = time.monotonic()+max(settings.TRIP_TIMEOUT_S+20,180)
            while time.monotonic() < deadline:
                job = client.get(f'/api/jobs/{job_id}').json()
                if job['status'] in ('done','failed'):
                    break
                time.sleep(.3)
            assert job['status'] == 'done', job.get('error')
            result = job['result']
            assert result['hmm_used'] and result['total_runtime_s'] < settings.TRIP_TIMEOUT_S
            records.append({'kind':kind,'trip_id':trip,'n_gps':int(inventory.set_index('trip_id').loc[trip,'n_gps']),
                            'n_arcs':int(inventory.set_index('trip_id').loc[trip,'n_arcs']),
                            'runtime_s':result['total_runtime_s'],'timings_s':result['timings_s'],
                            'peak_process_working_set_mb':peak_working_set_mb(),
                            'hmm_fallbacks':result['error_summary']['hmm_fallbacks'], 'warnings':result['warnings']})
    target = os.environ.get('PATHSNAP_VERIFY_DIR')
    if target:
        Path(target).mkdir(parents=True, exist_ok=True)
        (Path(target)/'train_stress.json').write_text(json.dumps(records,indent=2), encoding='utf-8')
