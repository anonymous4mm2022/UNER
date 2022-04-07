
The main code is based on the public work https://github.com/monologg/NER-Multimodal-pytorch

### Ablations

1. Text (BiLSTM-CRF)

2. Text+DNS
nohup python main.py --data_dir sample/ --model_dir model/text_dns --model_file model.pt --args_file args.pt --trainer_id 5 --preds_file results/text_dns_results.txt  --dns_feature_file dns_w2v_features.pt  --max_seq_len_dns 5 --dns_feat_dim 200 --gpu_idx 1  --do_train --do_eval > logs/text_dns.log &

3. Text+Image Global
nohup python main.py --data_dir sample/ --model_dir model/text_gImg --model_file model.pt --args_file args.pt --trainer_id 0 --preds_file results/text_gImg_results.txt  --img_feature_file img_resnet50_features.pt --num_img_region 49 --img_feat_dim 512 --gpu_idx 0  --do_train --do_eval > logs/text_gImg.log &


4. Text+Image Regions
nohup python main.py --data_dir sample/ --model_dir model/text_rImg --model_file model.pt --args_file args.pt --trainer_id 0 --preds_file results/text_rImg_results.txt  --img_feature_file img_frcn_features.pt --num_img_region 4 --img_feat_dim 512 --gpu_idx 1  --do_train --do_eval > logs/text_rImg.log &

5. Text+OCR Results
nohup python main.py --data_dir sample/ --model_dir model/text_oImg --model_file model.pt --args_file args.pt --trainer_id 0 --preds_file results/text_oImg_results.txt  --img_feature_file img_ocr_features.pt --num_img_region 10 --img_feat_dim 512 --gpu_idx 3  --do_train --do_eval > logs/text_oImg.log &

6. Text + Images
nohup python main.py --data_dir sample/ --model_dir model/text_Img --model_file model.pt --args_file args.pt --trainer_id 0 --preds_file results/text_Img_results.txt  --img_feature_file img_total_features.pt --num_img_region 63 --img_feat_dim 512 --gpu_idx 0  --do_train --do_eval > logs/text_img.log &


6. Text Domain and Images
nohup python main.py --data_dir sample/ --model_dir model/total --model_file model.pt --args_file args.pt --trainer_id 6 --preds_file results/without.txt  --img_feature_file img_total_features.pt --num_img_region 63 --img_feat_dim 512 --dns_feature_file dns_w2v_features.pt  --max_seq_len_dns 5 --dns_feat_dim 200 --gpu_idx 3  --do_train --do_eval > logs/without.log &


7. Text and Cross Features between Images and Domain
nohup python main.py --data_dir sample/ --model_dir model/total2 --model_file model.pt --args_file args.pt --trainer_id 7 --preds_file results/total_results2.txt  --img_feature_file img_total_features.pt --num_img_region 63 --img_feat_dim 512 --dns_feature_file dns_w2v_features.pt  --max_seq_len_dns 5 --dns_feat_dim 200 --gpu_idx 2  --do_train --do_eval > logs/with.log &


### Contrasts

1. ACN
nohup python main.py --data_dir sample/ --model_dir model/ACN --model_file model.pt --args_file args.pt --trainer_id 0 --preds_file results/acn_results.txt  --img_feature_file img_vgg_features.pt --num_img_region 49 --img_feat_dim 512 --gpu_idx 0  --do_train --do_eval > logs/acn.log & 

2. ACN-Bert
nohup python main.py --data_dir sample/ --model_dir model/ACN_bert --model_file model.pt --args_file args.pt --trainer_id 6 --preds_file results/acn_bert_results.txt  --img_feature_file img_vgg_features.pt --num_img_region 49 --img_feat_dim 512 --gpu_idx 0  --do_train --do_eval > logs/acn_bert.log &
