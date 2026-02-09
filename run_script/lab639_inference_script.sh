: <<'END_COMMENT'
all parser parameters
    # dataset
        csv_path
        train_csv
        val_csv
        test_csv
        result_path
        data_path
        exp_name
        output_name
        fold_num
        num_classes
        num_frames

    # training & inference
        mode
        model_path
        batch_size
        num_workers
        num_epochs
        learning_rate
        weight_decay
        optimizer
        validation_interval
        fusion_type
        motion_score
END_COMMENT

python main.py \
--csv_path ./data/lab639_fisheye/S009_9_actions_4_view_region \
--train_csv fisheye639_S009 \
--val_csv fisheye639_S009 \
--test_csv fisheye639_S009 \
--result_path ./result/lab639_fisheye/S009_lab639_r3d_backbone_9_actions_4_view_region_sum \
--data_path /home/bhchen/action_recognition/dataset/lab639_fisheye/processed_data_S009_820_616 \
--fold_num 3 \
--num_classes 9 \
--num_frames 16 \
--num_views 4 \
--mode test \
--model_path 'S009_lab639_r3d_backbone_9_actions_4_view_region_sum_xx-xx-xx_xxxx' \
--batch_size 2 \
--num_workers 4 \
--fusion_type sum \
--motion_score False