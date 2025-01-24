filenames=(agnewsdataraw-8000_trans_subst_10 StackOverflow\
 searchsnippets_trans_subst_10 biomedical_trans_subst_10\
 TS_trans_subst_10 T_trans_subst_10 S_trans_subst_10 tweet-original-order_trans_subst_10)
bertnames=(AG SO SS Bio G-TS G-T G-S Tweet)
DataSize=(8000 20000 12340 20000 11109 11108 11108 2472)
classNum=(4 20 8 20 152 152 152 89)
max_seq_length=(32 25 32 32 40 16 32 20)
max_train_steps=(15000 17000 15000 9000 0 5000 0 0)

id=(3)


# for i in ${id[*]}
# do
# echo "I am good at ${filenames[$i]} and ${bertnames[$i]}"

# CUDA_VISIBLE_DEVICES=0 python pretrain.py \
#     --train_file ./AugData/augmented-datasets/${filenames[$i]}.csv\
#     --model_name_or_path ./pretrained-models/distilbert-base-nli-stsb-mean-tokens\
#     --output_dir ./pretrained-models/${bertnames[$i]}\
#     --num_train_epochs 4000\
#     --per_device_train_batch_size 3072\
#     --save_steps 250\
#     --max_seq_length 32\
#     --save_total_limit 5\
#     --resume_from_checkpoint /home/calf/ssd/agents/RAG_demo/RSTC/pretrained-models/AG/checkpoint-10000

# done
# sleep 120m && echo 'begain runing'

for i in ${id[*]}
do
echo "I am good at ${filenames[$i]} and ${bertnames[$i]}"

CUDA_VISIBLE_DEVICES=7 python pretrain_transformers.py \
    --train_file ./AugData/augmented-datasets/${filenames[$i]}.csv\
    --validation_file ./AugData/augmented-datasets/${filenames[$i]}.csv\
    --model_name_or_path ./pretrained-models/distilbert-base-nli-stsb-mean-tokens\
    --output_dir ./pretrained-models/${bertnames[$i]}\
    --num_train_epochs 4000000 --max_train_steps ${max_train_steps[$i]}\
    --per_device_train_batch_size 2048\
    --per_device_eval_batch_size 2048\
    --checkpointing_steps 1000 --with_tracking\
    --max_seq_length ${max_seq_length[$i]}

done

    # --resume_from_checkpoint /home/calf/ssd/agents/RAG_demo/RSTC/pretrained-models/AG/checkpoint-10000\
