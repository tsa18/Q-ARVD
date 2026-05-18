export EXP_NAME="Q-ARVD_W4A8"
export Bit_W=4
export Bit_A=8

export DATA_PATH="prompts/vbench/all_dimension_extended.txt"
export GEN_NUM="946"
## infer
echo "infer"
echo "exp name:${EXP_NAME}"
CUDA_VISIBLE_DEVICES=0 python load_and_inference.py   \
    --config_path configs/self_forcing_dmd.yaml \
    --checkpoint_path checkpoints/self_forcing_dmd.pt  \
    --data_path ${DATA_PATH} \
    --use_ema --weight_bit ${Bit_W} --act_bit ${Bit_A} \
    --use_quant --test_name  vbench_${EXP_NAME}_W${Bit_W}A${Bit_A}  --quant_ckpt_path exp_logs/${EXP_NAME}-w${Bit_W}a${Bit_A}/quant_model.ckpt \
    --gen_num ${GEN_NUM}



