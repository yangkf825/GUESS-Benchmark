#!/usr/bin/env bash
# 一键运行全部算法在 Facebook100 或 Twitch 数据集
# 用法:
#   bash experiments/run_all_fb_twitch.sh twitch   ./data 5
#   bash experiments/run_all_fb_twitch.sh facebook ./data 3

DATASET=${1:-twitch}
DATA=${2:-./data}
RUNS=${3:-5}
OUT=./results

echo "======================================================"
echo " GNN-UQ-Bench: dataset=$DATASET  data_root=$DATA  runs=$RUNS"
echo "======================================================"

ARGS="--dataset $DATASET --data_root $DATA --runs $RUNS"

python experiments/run_ungnn_fb_twitch.py  $ARGS --save_dir $OUT/ungnn_fb_twitch
python experiments/run_gats_fb_twitch.py   $ARGS --save_dir $OUT/gats_fb_twitch
python experiments/run_cagcn_fb_twitch.py  $ARGS --save_dir $OUT/cagcn_fb_twitch
python experiments/run_calgnn_fb_twitch.py $ARGS --save_dir $OUT/calgnn_fb_twitch
python experiments/run_gpn_fb_twitch.py    $ARGS --save_dir $OUT/gpn_fb_twitch
python experiments/run_gduq_fb_twitch.py   $ARGS --save_dir $OUT/gduq_fb_twitch

echo "======================================================"
echo " Done. Results in $OUT/"
echo "======================================================"

python plot_results.py --data_dir $OUT --out_dir ./figures --dataset $DATASET
