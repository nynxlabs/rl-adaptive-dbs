#!/usr/bin/env python3
"""Run init-30hz and qat benchmarks with MATLAB engine error handling."""
import sys, os, time, json, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["RL_ADAPTIVE_DBS_ROOT"] = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["RL_ADAPTIVE_DBS_MATLAB_MODEL"] = os.path.join(os.environ["RL_ADAPTIVE_DBS_ROOT"], "reference-material", "KumaraveluEtAl2016")

from pathlib import Path
from envs.mehregan.env import MehreganEnv
from controllers.ddpg import evaluate, EvalConfig

VARIANTS = ["init-30hz", "qat"]
SEEDS = [0, 1, 2, 3, 4]
EVAL_STEPS = 5
CKPT_DIR = Path("artifacts/ddpg")
RESULTS_DIR = Path("results/mehregan_eval/runs")

def run_one(env, variant, seed):
    ckpt = CKPT_DIR / f"{variant}_train0.pt"
    if not ckpt.exists():
        return {"error": f"checkpoint not found: {ckpt}"}
    
    print(f"  {variant} seed={seed}...", flush=True)
    t0 = time.time()
    try:
        result = evaluate(
            env,
            ckpt,
            config=EvalConfig(seed=seed, eval_steps=EVAL_STEPS),
            protocol="mehregan_eval",
            variant=variant,
        )
        elapsed = time.time() - t0
        print(f"    done in {elapsed:.0f}s", flush=True)
        return result
    except Exception as e:
        elapsed = time.time() - t0
        print(f"    FAILED after {elapsed:.0f}s: {e}", flush=True)
        traceback.print_exc()
        return {"error": str(e)}

def main():
    print(f"=== Benchmark runner started at {time.strftime('%H:%M:%S')} ===", flush=True)
    print(f"Variants: {VARIANTS}, Seeds: {SEEDS}, Eval steps: {EVAL_STEPS}", flush=True)
    
    env = MehreganEnv()
    results = {}
    
    try:
        for variant in VARIANTS:
            print(f"\n--- {variant} ---", flush=True)
            for seed in SEEDS:
                result = run_one(env, variant, seed)
                key = f"{variant}_seed{seed}"
                results[key] = result
                
                # If MATLAB engine crashes, try to recover
                if "error" in result and "EngineError" in str(result.get("error", "")):
                    print("  MATLAB engine error detected, closing and recreating env...", flush=True)
                    try:
                        env.close()
                    except:
                        pass
                    time.sleep(5)
                    env = MehreganEnv()
                    print("  Env recreated, retrying...", flush=True)
                    result = run_one(env, variant, seed)
                    results[key] = result
    finally:
        try:
            env.close()
        except:
            pass
    
    # Summary
    successes = sum(1 for r in results.values() if "error" not in r)
    failures = sum(1 for r in results.values() if "error" in r)
    print(f"\n=== Summary: {successes} succeeded, {failures} failed ===", flush=True)
    for k, r in results.items():
        status = "OK" if "error" not in r else f"FAIL: {r['error'][:80]}"
        print(f"  {k}: {status}", flush=True)
    
    return 0 if failures == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
