export EXP_NAME="bfloat16"

export DATA_PATH="prompts/vbench/all_dimension_extended.txt"
export GEN_NUM="946"
## infer
echo "bfloat16 infer"
echo "exp name:${EXP_NAME}"
CUDA_VISIBLE_DEVICES=0 python load_and_inference.py   \
    --config_path configs/self_forcing_dmd.yaml \
    --checkpoint_path checkpoints/self_forcing_dmd.pt  \
    --data_path ${DATA_PATH} \
    --use_ema \
    --test_name  vbench_${EXP_NAME}  \
    --gen_num ${GEN_NUM}



