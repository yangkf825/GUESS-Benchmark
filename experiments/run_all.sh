#!/usr/bin/env bash
# 一键运行全部六种算法
# 用法:
#   bash experiments/run_all.sh elliptic ./elliptic 5
#   bash experiments/run_all.sh arxiv    ./data.pkl 5
#   bash experiments/run_all.sh eerm     ./cora     5 cora
#   bash experiments/run_all.sh eerm     ./amazon   5 amazon

DATASET=${1:-elliptic}
DATA=${2:-./elliptic}
RUNS=${3:-5}
EERM_DS=${4:-cora}
OUT=./results

echo "======================================================"
echo " GNN-UQ-Bench: dataset=$DATASET  runs=$RUNS"
echo "======================================================"

if [ "$DATASET" = "eerm" ]; then
    EXTRA="--eerm_dataset $EERM_DS --eerm_root $DATA"
    DS_ARG="--dataset eerm $EXTRA"
elif [ "$DATASET" = "arxiv" ]; then
    DS_ARG="--dataset arxiv --data_path $DATA"
else
    DS_ARG="--dataset elliptic --data_dir $DATA"
fi

python experiments/run_ungnn.py  $DS_ARG --runs $RUNS --save_dir $OUT/ungnn
python experiments/run_gats.py   $DS_ARG --runs $RUNS --save_dir $OUT/gats
python experiments/run_cagcn.py  $DS_ARG --runs $RUNS --save_dir $OUT/cagcn
python experiments/run_calgnn.py $DS_ARG --runs $RUNS --save_dir $OUT/calgnn
python experiments/run_gpn.py    $DS_ARG --runs $RUNS --save_dir $OUT/gpn
python experiments/run_gduq.py   $DS_ARG --runs $RUNS --save_dir $OUT/gduq

echo "======================================================"
echo " All done. Results in $OUT/"
echo "======================================================"

python plot_results.py --data_dir $OUT --out_dir ./figures --dataset $DATASET
