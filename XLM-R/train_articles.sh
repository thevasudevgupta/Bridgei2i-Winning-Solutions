#!/bin/bash
python run_glue.py \
--model_name_or_path xlm-roberta-base \
--do_train \
--do_eval \
--do_predict \
--test_file article_test.csv \
--train_file article_train.csv \
--validation_file article_test.csv \
--max_seq_length 128 \
--warmup_steps  100  \
--lr_scheduler_type linear \
--per_device_train_batch_size 32 \
--learning_rate 2e-5 \
--max_step 330 \
--num_train_epochs 3 \
--output_dir results_dir/