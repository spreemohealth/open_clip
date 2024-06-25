python -m training.main     --save-frequency 1  \
   --zeroshot-frequency 1     --train-data="/home/ubuntu/workspace/urg_data_prep/xray_aiqc_wrapper/xr_knee_train.pkl"\
         --val-data="/home/ubuntu/workspace/urg_data_prep/xray_aiqc_wrapper/xr_knee_val.pkl"      --dataset-type="mhddataset"\
              --warmup 50     --batch-size=128     --lr=1e-3     --wd=0.1     --epochs=200     --workers=32 \
                  --model resnet_pubmedbert     --precision "pure_bf16" --accum-freq 32 --report-to wandb --log-every-n-steps 1

python -m training.main     --save-frequency 1  --zeroshot-frequency 1     --train-data="/home/ubuntu/workspace/urg_data_prep/xray_aiqc_wrapper/xr_knee_train.pkl"\
 --val-data="/home/ubuntu/workspace/urg_data_prep/xray_aiqc_wrapper/xr_knee_val.pkl"      --dataset-type="mhddataset"\
 --warmup 350   --lock-text --batch-size=256     --lr=2e-4     --wd=0.1     --epochs=500     --workers=32 \
 --model resnet_pubmedbert     --precision "pure_bf16" --accum-freq 1 --report-to wandb --log-every-n-steps 1 --delete-previous-checkpoint

 -m training.main --save-frequency 1 --zeroshot-frequency 1 --train-data=/home/ubuntu/workspace/urg_data_prep/xray_aiqc_wrapper/xr_knee_train.pkl --val-data=/home/ubuntu/workspace/urg_data_prep/xray_aiqc_wrapper/xr_knee_val.pkl --dataset-type=mhddataset --warmup 180 --lock-text --batch-size=256 --lr=5e-4 --wd=0.1 --epochs=500 --workers=32 --model resnet_pubmedbert --precision pure_bf16 --accum-freq 1 --report-to wandb --log-every-n-steps 1

 #### latest
 python -m training.main --save-frequency=1  --zeroshot-frequency=1  \
    --train-data="/home/ubuntu/workspace/urg_data_prep/xray_aiqc_wrapper/final_df_for_dls/xr_knee_office_train.pkl" \
     --val-data="/home/ubuntu/workspace/urg_data_prep/xray_aiqc_wrapper/final_df_for_dls/xr_knee_office_val.pkl" \
     --dataset-type="mhddataset" --warmup=350  --batch-size=256 --lr=2e-4     --wd=0.1     --epochs=150     --workers=32 \
       --model resnet_pubmedbert_reduce_dim --precision "pure_bf16" --accum-freq 1 --log-every-n-steps 1 --delete-previous-checkpoint \
        --report-to wandb 

### grad accum to 4096 
python -m training.main     --save-frequency 1  --zeroshot-frequency 1     --train-data="/home/ubuntu/workspace/urg_data_prep/xray_aiqc_wrapper/xr_knee_train.pkl" --val-data="/home/ubuntu/workspace/urg_data_prep/xray_aiqc_wrapper/xr_knee_val.pkl"      --dataset-type="mhddataset" --warmup 22   --lock-text --batch-size=256     --lr=1e-4     --wd=0.1     --epochs=150     --workers=32  --model resnet_pubmedbert     --precision "pure_bf16" --accum-freq 16 --log-every-n-steps 1 --delete-previous-checkpoint --report-to wandb

### latest 100k run with resnet_pubmedbert
python -m training.main --save-frequency 1  --zeroshot-frequency 1   --train-data="/home/ubuntu/workspace/urg_data_prep/xray_aiqc_wrapper/final_df_for_dls/XR_knee_all_train_v2.pkl"  --val-data="/home/ubuntu/workspace/urg_data_prep/xray_aiqc_wrapper/final_df_for_dls/XR_knee_all_val.pkl"    --dataset-type="mhddataset" --warmup 350  --batch-size=128 --lr=1e-4     --wd=0.2     --epochs=75  --workers=32    --model resnet_pubmedbert_reduce_dim --precision "pure_bf16" --accum-freq 2 --log-every-n-steps 1 --delete-previous-checkpoint     --report-to wandb

### save with metric
###
python -m training.main --save-frequency 1 --zeroshot-frequency 1 --train-data="/home/ubuntu/workspace/urg_data_prep/xray_aiqc_wrapper/final_df_for_dls/XR_Knee_all_studies_df_for_dl_V2_num_views<=7_p_splits_train.pkl" --val-data="/home/ubuntu/workspace/urg_data_prep/xray_aiqc_wrapper/final_df_for_dls/XR_Knee_all_studies_df_for_dl_V2_num_views<=7_p_splits_val.pkl" --dataset-type=mhddataset --warmup 350 --batch-size=128 --lr=1e-4 --wd=0.2 --epochs=75 --workers=32 --model resnet_pubmedbert_reduce_dim --precision pure_bf16 --accum-freq 2 --log-every-n-steps 1 --delete-previous-checkpoint --report-to wandb --image-mean 0 --image-std 1


#### test view agnostic 
python -m training.main --save-frequency 1  --zeroshot-frequency 1   --train-data="/home/ubuntu/workspace/urg_data_prep/xray_aiqc_wrapper/final_df_for_dls/XR_knee_all_train_v2.pkl"  --val-data="/home/ubuntu/workspace/urg_data_prep/xray_aiqc_wrapper/final_df_for_dls/XR_knee_all_val.pkl"    --dataset-type="mhddataset" --warmup 350  --batch-size=128 --lr=1e-4     --wd=0.2     --epochs=0  --workers=32    --model resnet_pubmedbert_reduce_dim --precision "pure_bf16" --accum-freq 2 --log-every-n-steps 1 --delete-previous-checkpoint     --report-to wandb --resume /home/ubuntu/workspace/multimodal/open_clip/src/logs/2024_05_17-10_22_52-model_resnet_pubmedbert_reduce_dim-lr_0.0001-b_128-j_32-p_pure_bf16/best_24.pt