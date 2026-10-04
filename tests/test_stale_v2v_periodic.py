import pytest
from platooning.platooning.dynamic_manager import DynamicPlatoonManager
from platooning.models.vehicle import VehicleState
import time

def test_stale_v2v_periodic_no_recreate():
    config = {
        'communication': {'stale_threshold_ms': 200.0},
        'platooning': {
            'enabled': True,
            'max_formation_distance_m': 50.0,
            'max_speed_difference_mps': 3.0,
            'minimum_platoon_size': 2,
            'maximum_platoon_size': 5,
        },
        'dynamic': {
            'reconfiguration_interval_ms': 2000,
            'reconfiguration_cooldown_ms': 1000,
            'minimum_improvement': 0.5,
        },
        'qubo': {'enabled': True, 'solver': 'exact'},
    }

    dm = DynamicPlatoonManager(config, optimizer_mode='exact_qubo')
    states = [
        VehicleState('V1', 0.0, 100.0, 0.0, 30.0, 'r1', 'l1'),
        VehicleState('V2', 0.0, 95.0, 0.0, 30.0, 'r1', 'l1'),
    ]
    latest_known = {s.vehicle_id: {s.vehicle_id: s} for s in states}
    
    # Initial formation -> P1=[V1,V2]
    dm.update_vehicle_states(latest_known, 0.0)
    assert len(dm.platoons) == 1
    assert dm.platoons[0].size == 2
    
    # 0.3s elapses, V2 state is now stale (> 200ms)
    dm.update_vehicle_states(latest_known, 0.3)
    # Stale state triggers critical violation and drops platoon
    assert len(dm.platoons) == 0
    
    # 2.5s elapses (triggering PERIODIC_REOPTIMIZATION) but V2 state STILL stale
    dm.update_vehicle_states(latest_known, 2.5)
    # Platoon MUST NOT be recreated
    assert len(dm.platoons) == 0

