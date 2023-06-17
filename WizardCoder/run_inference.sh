#!/usr/bin/env bash
set -x

python3 src/inference_wizardcoder.py \
    --base_model "/dataset/home/liaoxingyu/models/WizardCoder-15B-V1.0" \
    --input_data_path "data.jsonl" \
    --output_data_path "result.jsonl"
