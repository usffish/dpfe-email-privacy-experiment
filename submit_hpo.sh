#!/bin/bash
# Submit N parallel HPO jobs to CIRCE.
# Each job runs one trial and writes results to the shared SQLite study DB.
#
# Usage:
#   bash submit_hpo.sh        # submit 8 jobs (default)
#   bash submit_hpo.sh 16     # submit 16 jobs
#   bash submit_hpo.sh 4 attack-training-hpo-v2   # custom study name

N=${1:-8}
STUDY=${2:-"attack-hpo-v2"}

echo "Submitting $N HPO jobs (study: $STUDY)"

for i in $(seq 1 $N); do
    sbatch --export=ALL,HPO_STUDY_NAME="$STUDY" run_hpo.sbatch
done

echo ""
echo "Monitor progress:"
echo "  python view_hpo.py --study $STUDY"
echo ""
echo "Submit more trials later:"
echo "  bash submit_hpo.sh 8 $STUDY"
