"""
Reproduce paper results (Experiment 1 and 2).
"""
import sys
sys.path.append('..')

from core_engine import DREngine, State

def run_experiment_1():
    start = State((1,1,1,1,1,1))
    target = State((0,1,0,1,1,1))
    engine = DREngine(start)
    engine.apply_sbf(0)
    engine.apply_sbf(2)
    engine.apply_brp()
    return engine.get_state() == target

def run_experiment_2():
    start = State((0,0,0,0,0,0))
    target = State((1,0,1,0,1,1))
    engine = DREngine(start)
    engine.apply_gi()
    engine.apply_mr((1,0,1,0))
    return engine.get_state() == target

print("Experiment 1:", "PASS" if run_experiment_1() else "FAIL")
print("Experiment 2:", "PASS" if run_experiment_2() else "FAIL")
