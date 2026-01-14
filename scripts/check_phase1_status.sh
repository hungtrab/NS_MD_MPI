#!/bin/bash
# Check status of Phase 1 parallel experiments

if [ ! -f logs/phase1/pids.txt ]; then
    echo "❌ No running experiments found (pids.txt missing)"
    exit 1
fi

pids=($(cat logs/phase1/pids.txt))
names=(
    "1.1_vanilla"
    "1.2_moderate_baseline"
    "1.3_moderate_nsmdmpi"
)

echo "================================================================"
echo "  PHASE 1 STATUS CHECK"
echo "================================================================"
echo ""

running=0
completed=0
failed=0

for i in "${!pids[@]}"; do
    pid="${pids[$i]}"
    name="${names[$i]}"
    logfile="logs/phase1/${name}.log"
    
    echo "[$((i+1))/3] $name (PID: $pid)"
    
    if ps -p $pid > /dev/null 2>&1; then
        echo "  Status: ✅ RUNNING"
        running=$((running + 1))
        
        # Show last few lines of log
        if [ -f "$logfile" ]; then
            echo "  Last update:"
            tail -n 3 "$logfile" | sed 's/^/    /'
        fi
    else
        # Check exit code from log
        if [ -f "$logfile" ]; then
            if grep -q "Final episodic reward" "$logfile"; then
                echo "  Status: ✅ COMPLETED"
                completed=$((completed + 1))
            else
                echo "  Status: ❌ FAILED"
                failed=$((failed + 1))
                echo "  Check log: $logfile"
            fi
        else
            echo "  Status: ❓ UNKNOWN (no log)"
        fi
    fi
    echo ""
done

echo "================================================================"
echo "Summary:"
echo "  Running:   $running"
echo "  Completed: $completed"
echo "  Failed:    $failed"
echo "================================================================"
