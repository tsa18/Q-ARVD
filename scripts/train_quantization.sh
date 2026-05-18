export EXP_NAME="Q-ARVD_W4A8"
export Bit_W=4
export Bit_A=8


echo "exp name: ${EXP_NAME}"
echo "train quantization"
### --sensitivity <chunkwise sensitivity obtained in Step1>
### --weight_bit <weight bitwith> --act_bit <activation bitwidth>
# train
CUDA_VISIBLE_DEVICES=0 python quant_self_forcing.py   \
    --config_path configs/self_forcing_dmd.yaml \
    --checkpoint_path  checkpoints/self_forcing_dmd.pt \
    --data_path prompts/MovieGenVideoBench_extended.txt \
    --sensitivity "0.5462,0.1668,0.1189,0.0789,0.0534,0.0263,0.0096" \
    --use_ema --weight_bit ${Bit_W} --act_bit ${Bit_A} --exp_name  ${EXP_NAME} 
