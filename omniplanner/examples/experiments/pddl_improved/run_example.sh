#!/usr/bin/env bash
set -euo pipefail

OUTPUT_DIR="/home/user/workspace/output/batch_results"

echo "============================================================"
echo " DSG-PDDL batch experiment"
echo "============================================================"
echo
echo "Output directory: ${OUTPUT_DIR}"
echo
echo "Experiment grid:"
echo "  num_nodes:         100 200 400 800"
echo "  num_objects:       50"
echo "  num_regions:       4"
echo "  goal_conjunctions: 6"
echo "  goal_seeds:        1 2 3 4 5"
echo "  pddl_domain:       explicit"
echo "  pddl_sampler:      compressed"
echo "  pddl_solver:       lazy_ff"
echo
echo "Expected runs: 4 node values × 5 goal seeds = 20"
echo "Existing completed runs will be skipped."
echo

python run_batch_experiments.py \
  --script run_experiment.py \
  --output-dir "${OUTPUT_DIR}" \
  --num-nodes 100 200 400 800 \
  --num-objects 50 \
  --num-regions 4 \
  --goal-conjunctions 6 \
  --goal-seeds 1 2 3 4 5 \
  --pddl-domains explicit \
  --pddl-samplers compressed \
  --pddl-solvers lazy_ff \
  --skip-existing

echo
echo "============================================================"
echo " Batch execution finished"
echo "============================================================"
echo
echo "Opening interactive aggregated results viewer..."
echo
echo "Select 'num_nodes' when asked for the varying parameter."
echo

python print_batch_results.py \
  --batch-dir "${OUTPUT_DIR}"

echo
echo "Done."