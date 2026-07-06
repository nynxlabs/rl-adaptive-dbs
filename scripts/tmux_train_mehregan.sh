#!/bin/bash
# Run init-30hz training with MATLAB backend in tmux session
# Paper training worked with MATLAB — init-30hz should too
# The earlier hangs were likely caused by the process getting killed, not a MATLAB bug

cd /home/nynxbox/neuroengineering/rl-adaptive-dbs
source scripts/matlab/env.sh

echo "=== init-30hz training started: $(date) ==="
echo "Plant backend: matlab (default)"

# Clean stale log
rm -f artifacts/ddpg/train_init-30hz_seed0.log

# Run training
uv run rl-dbs train --controller ddpg --variant init-30hz --seeds 0 --episodes 10 2>&1

echo "=== init-30hz training finished: $(date) ==="

# Check checkpoint
if [ -f artifacts/ddpg/init-30hz_train0.pt ]; then
    echo "✓ checkpoint created: $(ls -la artifacts/ddpg/init-30hz_train0.pt)"
    
    # Train qat
    echo ""
    echo "=== qat training started: $(date) ==="
    uv run rl-dbs train --controller ddpg --variant qat --seeds 0 --episodes 10 2>&1
    echo "=== qat training finished: $(date) ==="
    
    if [ -f artifacts/ddpg/qat_train0.pt ]; then
        echo "✓ qat checkpoint created"
        
        # Run remaining benchmarks
        echo ""
        echo "=== Running remaining benchmarks ==="
        for variant in init-30hz qat; do
            count=$(find results/mehregan_eval/runs -maxdepth 1 -name "ddpg_${variant}_*" -type d | wc -l)
            if [ "$count" -ge 5 ]; then
                echo "  ddpg:$variant already has $count runs, skipping."
            else
                echo "  Running ddpg:$variant (5 seeds)..."
                uv run rl-dbs benchmark --suite-name mehregan_eval --controllers "ddpg:$variant" --seeds 0,1,2,3,4 --no-timeseries 2>&1
            fi
        done
        
        # Summary
        echo ""
        echo "=== Generating summary ==="
        uv run rl-dbs summary --suite-name mehregan_eval 2>&1
        uv run rl-dbs summary --suite-name mehregan_eval --csv results/mehregan_eval/summary.csv 2>&1
        
        TOTAL=$(find results/mehregan_eval/runs -maxdepth 1 -type d | tail -n +2 | wc -l)
        echo ""
        echo "=== COMPLETE: $TOTAL/40 runs at $(date) ==="
    else
        echo "ERROR: qat checkpoint not created"
    fi
else
    echo "ERROR: init-30hz checkpoint not created"
fi
