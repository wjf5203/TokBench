#!/bin/bash

tokenizer_name="cogvideox"
recon_result="reconstructions_videos"
resolution=256

dataset_names=("ch3" "ds")
for dataset_name in "${dataset_names[@]}"
do
  CUDA_VISIBLE_DEVICES=3 python eval_text.py  \
  --img_folder "${recon_result}/${tokenizer_name}/text_data/${dataset_name}_${resolution}/" \
  --gt_path "TokBench/video_annotations/text_${dataset_name}.json" \
  --dataset "${dataset_name}" \
  --data_type "video" \
  --batch_size 64 \
  --method_name $tokenizer_name \
  --setting $resolution  \
  --save_dir  video_outputs  &\
done



CUDA_VISIBLE_DEVICES=4 python eval_face.py  \
    --original_image_path TokBench/videos/face_data/face_clip_3s \
    --reconstruction_image_path "${recon_result}/${tokenizer_name}/face_data/face_clip_3s_${resolution}/" \
    --tokenizer $tokenizer_name \
    --data_type "video" \
    --meta_path TokBench/video_annotations/videoface_meta.json \
    --setting $resolution \
    --save_dir  video_outputs  

python compute_all_metrics.py  --setting $resolution --data_type video  --output_path  video_outputs
