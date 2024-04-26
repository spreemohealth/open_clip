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
 python -m training.main     --save-frequency 1  --zeroshot-frequency 1  \
    --train-data="/home/ubuntu/workspace/urg_data_prep/xray_aiqc_wrapper/xr_knee_train.pkl"\
     --val-data="/home/ubuntu/workspace/urg_data_prep/xray_aiqc_wrapper/xr_knee_val.pkl"\
         --dataset-type="mhddataset" --warmup 350   --lock-text --batch-size=256\
              --lr=2e-4     --wd=0.1     --epochs=150     --workers=32  --model resnet_pubmedbert\
                   --precision "pure_bf16" --accum-freq 1 --log-every-n-steps 1 --delete-previous-checkpoint\
                    --report-to wandb

### grad accum to 4096 
python -m training.main     --save-frequency 1  --zeroshot-frequency 1     --train-data="/home/ubuntu/workspace/urg_data_prep/xray_aiqc_wrapper/xr_knee_train.pkl" --val-data="/home/ubuntu/workspace/urg_data_prep/xray_aiqc_wrapper/xr_knee_val.pkl"      --dataset-type="mhddataset" --warmup 22   --lock-text --batch-size=256     --lr=1e-4     --wd=0.1     --epochs=150     --workers=32  --model resnet_pubmedbert     --precision "pure_bf16" --accum-freq 16 --log-every-n-steps 1 --delete-previous-checkpoint --report-to wandb