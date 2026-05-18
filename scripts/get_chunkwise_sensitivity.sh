CUDA_VISIBLE_DEVICES=0 python get_chunk_wise_weights.py   \
    --config_path configs/self_forcing_dmd.yaml \
    --checkpoint_path checkpoints/self_forcing_dmd.pt \
    --data_path prompts/MovieGenVideoBench_extended.txt  \
    --use_ema --weight_bit 4 --act_bit 8 --exp_name Get_chunk_wise_weights_w4a8 # set the quantization bitwidth and exp log name