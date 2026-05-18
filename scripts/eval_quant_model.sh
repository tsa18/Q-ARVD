export EXP_NAME="Q-ARVD_W4A8"
export Bit_W=4
export Bit_A=8

## eval 
echo "eval"
# FVD-FP and LPIPS-FP
python common_metrics_on_video_quality/cal_metrics_by_folder_efficient.py --gt_videos_dir test_out/vbench_bfloat16 --gen_videos_dir test_out/vbench_${EXP_NAME}_W${Bit_W}A${Bit_A}
# Vbench
vbench evaluate \
    --dimension "subject_consistency background_consistency motion_smoothness aesthetic_quality imaging_quality" \
    --videos_path test_out/vbench_${EXP_NAME}_W${Bit_W}A${Bit_A} \
    --mode=custom_input