#!/bin/bash
# Submit N parallel HPO jobs to CIRCE.
# Each job runs one trial and writes results to the shared JournalFile study DB.
#
# Usage:
#   bash submit_hpo.sh                              # 8 jobs, gpt2 (default)
#   bash submit_hpo.sh 16                           # 16 jobs, gpt2
#   bash submit_hpo.sh 8 attack-hpo-v4              # custom study name, gpt2
#   bash submit_hpo.sh 8 gpt-neo-hpo-v1 run_hpo_gptneo.sbatch  # gpt-neo

N=${1:-8}
STUDY=${2:-"attack-hpo-v4"}
SBATCH_FILE=${3:-"run_hpo.sbatch"}

echo "Submitting $N HPO jobs (study: $STUDY, sbatch: $SBATCH_FILE)"

for i in $(seq 1 $N); do
    sbatch --export=ALL,HPO_STUDY_NAME="$STUDY" "$SBATCH_FILE"
done

echo ""
echo "Monitor progress:"
echo "  python view_hpo.py --study $STUDY"
echo ""
echo "Submit more trials later:"
echo "  bash submit_hpo.sh 8 $STUDY $SBATCH_FILE"
