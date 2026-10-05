#!/usr/bin/env bash

set -euo pipefail

: "${DATA_ROOT:?Set DATA_ROOT to the directory containing MIntRec.}"
GPU_ID="${GPU_ID:-0}"

for seed in 0 1 2 3 4
do
    for multimodal_method in 'mcsp'
    do
        for method in 'mcsp'
        do
            for text_backbone in 'bert-base-uncased'
            do
                for dataset in 'MIntRec'
                do
                    python run.py \
                    --dataset $dataset \
                    --data_path "$DATA_ROOT" \
                    --logger_name $method \
                    --multimodal_method $multimodal_method \
                    --method $method\
                    --train \
                    --tune \
                    --save_results \
                    --seed $seed \
                    --gpu_id "$GPU_ID" \
                    --video_feats_path 'swin_feats.pkl' \
                    --audio_feats_path 'wavlm_feats.pkl' \
                    --text_backbone $text_backbone \
                    --config_file_name ${method}_${dataset} \
                    --results_file_name "mintrec_mcsp.csv" \
                    --output_path "outputs/${dataset}"
                done
            done
        done
    done
done
