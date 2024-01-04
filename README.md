# ShortcutLearnability
This is the official implementation of our paper "Which Shortcut Solution Do Question Answering Models Prefer to Learn?" (Kazutoshi Shinoda, Saku Sugawara, Akiko Aizawa) at AAAI-23.

## 0. Environments and Setups
- torch==1.10
- transformers==4.18.0

The used random seeds were 42, 43, 44, 45, and 46.

We basically used the same hyperparameters as the original papers.

## 1. Defining shortcut and anti-shortcut examples
- Extractive QA
  - analyze_datasets.py

      ```
      data="${SQuAD_DIR}/train-v1.1.json"
      n_workers="4"
      nohup python -u analyze_datasets.py --data_path ${data} --n_workers ${n_workers} --do_light --analyses answer-position-sentence question-context-ngram-overlap-per-sent question-context-similar-sent answer-candidates > log/analysis &
      ```

- Multiple-choice QA
  - PreviewQAExamples&BiasAnalysis.ipynb

## 2. Behavioral Tests: Learning from Biased Training Sets
- Training and Evaluation
  - Extractive QA

      ```
      SEED="42"
      GPU_ID="0"
      RUN_NAME="bert_squad_3d-biased-aps-qcss-ac_seed${SEED}"
      PJ_NAME="exqa-squad"
      CUDA_VISIBLE_DEVICES=$GPU_ID nohup python -u run_squad.py --project $PJ_NAME \
      --model_type bert \
      --model_name_or_path bert-base-uncased --do_lower_case \
      --do_train --do_eval --output_dir $RE_EXQA_OUT_DIR/$RUN_NAME --warmup_ratio 0.1 --num_train_epochs 10 --save_steps 200 --logging_train_steps 50 --log_before_train --evaluate_during_training --overwrite_output_dir --threads 4 --do_biased_train \
      --bias_1 answer-position-sentence \
      --bias_1_included_in 0 \
      --bias_2 question-context-similar-sent \
      --bias_2_included_in 0 \
      --bias_3 answer-candidates \
      --bias_3_included_in 1 \
      --train_file $SQuAD_DIR/train-v1.1.json \
      --predict_file $SQuAD_DIR/dev-v1.1.json \
      --seed $SEED > log/$RUN_NAME &
      ```

  - Multiple-choice QA

      ```
      SEED="42"
      RUN_NAME="bert_race_biased-maxlo-1-top50-1_seed${SEED}"
      CUDA_VISIBLE_DEVICES=1 WANDB_PROJECT="mcqa-race" nohup python -u run_multiple_choice.py \
      --task_name race --model_name_or_path bert-base-uncased \
      --bias_1 correct-has-max-lexical-overlap \
      --bias_1_included_in 1 \
      --bias_2 only-correct-has-top50-words \
      --bias_2_included_in 1 \
      --do_biased_train --do_train --do_eval --do_predict \
      --predict_all_checkpoints --data_dir $RACE_DIR \
      --learning_rate 1e-5 --num_train_epochs 10 --max_seq_length 512 \
      --output_dir $RE_MCQA_OUT_DIR/$RUN_NAME \
      --per_device_eval_batch_size 16 \
      --per_device_train_batch_size 8 \
      --gradient_accumulation_steps 4 \
      --max_grad_norm 1 \
      --adam_beta1 0.9 \
      --adam_beta2 0.98 \
      --adam_epsilon 1e-6 \
      --warmup_ratio 0.06 \
      --weight_decay 0.01 \
      --logging_steps 10 \
      --save_steps 100 \
      --eval_steps 100 \
      --evaluate_during_training \
      --seed $SEED \
      --overwrite_output > log/$RUN_NAME &
      ```

- Results
  - Biased-AntiBiased-Evaluation.ipynb

## 3. Visualizing the Loss Landscape
- Experiments (This will take few days.)
  - Training

  ```
  SEED="42"
  RUN_NAME="bert_squad_vis-aps_1400-ex_seed42"
  PJ_NAME="exqa-squad"
  CUDA_VISIBLE_DEVICES=0 nohup python -u run_squad.py --project $PJ_NAME --model_type bert --model_name_or_path bert-base-uncased --do_lower_case --do_train --do_eval --do_fewshot_train --num_fewshot_examples 1400 --output_dir $RE_EXQA_OUT_DIR/$RUN_NAME --warmup_ratio 0.1 --num_train_epochs 10 --logging_train_steps 1000 --evaluate_during_training --overwrite_output_dir --threads 4 --do_biased_train --bias_1 answer-position-sentence --bias_1_included_in 0 --bias_2 question-context-similar-sent --bias_2_not_equal answer-position-sentence --bias_3 answer-candidates --bias_3_larger_than 2 --train_file $SQuAD_DIR/train-v1.1.json --predict_file $SQuAD_DIR/dev-v1.1.json --seed $SEED > log/$RUN_NAME &
  ```

  - Computing the surface

  ```
  MODEL_ID="bert_squad_vis-aps_1400-ex_seed42"
  PLOT_ID="bert_squad"
  WIDTH="101"
  CUDA_VISIBLE_DEVICES=2 nohup python -u plot_surface.py \
  --plot_id $PLOT_ID \
  --task_type ex-qa \
  --task_name squad \
  --surface_id ${MODEL_ID}_width-${WIDTH} \
  --base_model_path $RE_EXQA_OUT_DIR/bert_squad \
  --model_path $RE_EXQA_OUT_DIR/$MODEL_ID \
  --batch_size 256 \
  --width $WIDTH \
  --do_setup \
  --do_random_plot > log/${MODEL_ID}_width-${WIDTH} &
  ```

- Visualization
  - Please download ParaView for surface visualization from [the official site](https://www.paraview.org/).

## 4. Rissanen Shortcut Analysis
- Training and Evaluation
  - Extractive QA

  ```
  PJ_NAME="exqa-squad"
  SEED="42"
  GPU_ID="0"
  NUM_FEWSHOT="1400"
  KEY="ex-long"
  RUN_NAME="bert_squad_mdl-aps_${KEY}_seed${SEED}"
  CUDA_VISIBLE_DEVICES=${GPU_ID} nohup python -u run_squad.py --project $PJ_NAME --model_type bert \
  --model_name_or_path bert-base-uncased --do_lower_case --do_train \
  --do_online_code --do_fewshot_train --num_fewshot_examples $NUM_FEWSHOT \
  --do_fewshot_unique_features --do_exclude_long_context \
  --seed $SEED --output_dir $RE_EXQA_OUT_DIR/MDL/$RUN_NAME \
  --warmup_ratio 0.1 --overwrite_output_dir --threads 4 --do_biased_train \
  --bias_1 answer-position-sentence --bias_1_included_in 0 \
  --bias_2 question-context-similar-sent --bias_2_not_equal answer-position-sentence \
  --bias_3 answer-candidates --bias_3_larger_than 2 \
  --train_file $SQuAD_DIR/train-v1.1.json --predict_file $SQuAD_DIR/dev-v1.1.json --seed $SEED > log/$RUN_NAME &
  ```

  - Multiple-choice QA
    - run_multiple_choice.py

- Results
  - RissanenDataAnalysis.ipynb

## 5. Balancing Shortcut and Anti-shortcut Examples
- Training and Evaluation
  - Extractive QA

  ```
  SEED="42"
  GPU_ID="1"
  RATIO="0.8"
  RUN_NAME="bert_squad_1d-blend-aps-${RATIO}_5k-ex_seed${SEED}"
  PJ_NAME="exqa-squad"
  CUDA_VISIBLE_DEVICES=$GPU_ID nohup python -u run_squad.py --project $PJ_NAME \
  --model_type bert \
  --model_name_or_path bert-base-uncased --do_lower_case \
  --do_train --do_eval --output_dir $RE_EXQA_OUT_DIR/$RUN_NAME --warmup_ratio 0.1 --num_train_epochs 10 --logging_train_steps 1000 --save_steps 1000 --num_total_examples 5000 --overwrite_output_dir --threads 4 \
  --do_biased_train \
  --bias_1 answer-position-sentence \
  --bias_1_included_in 0 \
  --bias_2 answer-candidates \
  --bias_2_larger_than 1 \
  --do_blend_anti_biased \
  --anti_biased_ratio $RATIO \
  --anti_bias_1 answer-position-sentence \
  --anti_bias_1_larger_than 1 \
  --anti_bias_2 answer-candidates \
  --anti_bias_2_larger_than 1 \
  --train_file $SQuAD_DIR/train-v1.1.json \
  --predict_file $SQuAD_DIR/dev-v1.1.json \
  --seed $SEED > log/$RUN_NAME &
  ```

  - Multiple-choice QA
    - run_multiple_choice.py

- Results
  - Biased-AntiBiased-Evaluation.ipynb

# Citation
If you find our codes useful, please cite our paper.

```
@article{Shinoda_Sugawara_Aizawa_2023,
  title={Which Shortcut Solution Do Question Answering Models Prefer to Learn?},
  volume={37},
  url={https://ojs.aaai.org/index.php/AAAI/article/view/26590},
  DOI={10.1609/aaai.v37i11.26590},
  number={11},
  journal={Proceedings of the AAAI Conference on Artificial Intelligence},
  author={Shinoda, Kazutoshi and Sugawara, Saku and Aizawa, Akiko},
  year={2023},
  month={Jun.},
  pages={13564-13572}
}
```

# Contact
Please feel free to contact me if you have any suggestions or questions.

Email: kazutoshi.shinoda0516@gmail.com / X(Twitter): [@shino__c](https://twitter.com/shino__c)
