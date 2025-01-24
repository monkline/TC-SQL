filenames=(agnewsdataraw-8000_trans_subst_10 stackoverflow_trans_subst_10\
 searchsnippets_trans_subst_10 biomedical_trans_subst_10\
 TS_trans_subst_10 T_trans_subst_10 S_trans_subst_10 tweet-original-order_trans_subst_10)
bertnames=(AG SO SS Bio G-TS G-T G-S Tweet)
DataSize=(8000 20000 12340 20000 11109 11108 11108 2472)
classNum=(4 20 8 20 152 152 152 89)
maxLen=(32 25 32 45 40 16 32 20)
M=(110 50 74 50 82 82 82 110)
pre_step=(60 -1 600 -1 600 600 600 -1)
reg2=(1 1 0.1 1 0.1 0.1 0.1 0.1)
logalpha=(10 10 10 10 100 100 100 100)
# epoch=100

id=(3)

for i in ${id[*]}
do
echo "I am good at ${filenames[$i]} and ${bertnames[$i]}"
CUDA_VISIBLE_DEVICES=1 python main.py\
    --bert ${bertnames[$i]}\
    --dataname ${filenames[$i]}\
    --num_classes ${classNum[$i]}\
    --classes ${classNum[$i]}\
    --max_length ${maxLen[$i]}\
    --M ${M[$i]} --epsion 0.1\
    --pre_step ${pre_step[$i]}\
    --reg2 ${reg2[$i]} --logalpha ${logalpha[$i]}

done





# CUDA_VISIBLE_DEVICES=0,1 python main.py --dataname agnewsdataraw-8000_trans_subst_10 --num_classes 4 --classes 4 --M 110 --epsion 0.1 --pre_step 600 --temperature 1 --start 0 --reg2 1 --logalpha 10 
# wait
# CUDA_VISIBLE_DEVICES=0,1 python main.py --dataname stackoverflow_trans_subst_10 --num_classes 20 --classes 20 --M 50 --epsion 0.1 --pre_step -1 --temperature 1 --start 0 --reg2 1 --logalpha 10 
# wait
# CUDA_VISIBLE_DEVICES=0,1 python main.py --dataname searchsnippets_trans_subst_10  --num_classes 8 --classes 8 --M 74 --epsion 0.1 --pre_step 600 --temperature 1 --start 0 --reg2 0.1  --logalpha 10
# wait
# CUDA_VISIBLE_DEVICES=0,1 python main.py --dataname biomedical_trans_subst_10 --num_classes 20 --classes 20 --M 50 --epsion 0.1 --pre_step -1 --temperature 1 --start 0  --reg2 1 --logalpha 10 
# wait
# CUDA_VISIBLE_DEVICES=0,1 python main.py --dataname TS_trans_subst_10 --num_classes 152 --classes 152 --M 82 --epsion 0.1 --pre_step 600 --temperature 1 --start 0  --reg2 0.1 --logalpha 100 
# wait
# CUDA_VISIBLE_DEVICES=0,1 python main.py --dataname T_trans_subst_10 --num_classes 152 --classes 152 --M 82 --epsion 0.1 --pre_step 600 --temperature 1 --start 0  --reg2 0.1 --logalpha 100 
# wait
# CUDA_VISIBLE_DEVICES=0,1 python main.py --dataname S_trans_subst_10 --num_classes 152 --classes 152 --M 82 --epsion 0.1 --pre_step 600 --temperature 1 --start 0 --reg2 0.1 --logalpha 100 
# wait
# CUDA_VISIBLE_DEVICES=0,1 python main.py --dataname tweet-original-order_trans_subst_10 --num_classes 89 --classes 89 --M 110  --epsion 0.1 --pre_step -1 --temperature 1 --start 0 --reg2 0.1 --logalpha 100 




# # # python main.py --dataname agnewsdataraw-8000_trans_subst_10 --num_classes 4 --classes 4 --M 110 --epsion 0.1 --pre_step 600 --temperature 1 --start 0 --reg2 1 --logalpha 10 --objective $1
# # # wait
# # # python main.py --dataname stackoverflow_trans_subst_10 --num_classes 20 --classes 20 --M 50 --epsion 0.1 --pre_step -1 --temperature 1 --start 0 --reg2 1 --logalpha 10 --objective $1
# # # wait
# # # python main.py --dataname searchsnippets_trans_subst_10  --num_classes 8 --classes 8 --M 74 --epsion 0.1 --pre_step 600 --temperature 1 --start 0 --reg2 0.1  --logalpha 10 --objective $1
# # # wait
# # # python main.py --dataname biomedical_trans_subst_10 --num_classes 20 --classes 20 --M 50 --epsion 0.1 --pre_step -1 --temperature 1 --start 0  --reg2 1 --logalpha 10 --objective $1
# # # wait
# # # python main.py --dataname TS_trans_subst_10 --num_classes 152 --classes 152 --M 82 --epsion 0.1 --pre_step 600 --temperature 1 --start 0  --reg2 0.1 --logalpha 100 --objective $1
# # # wait
# # # python main.py --dataname T_trans_subst_10 --num_classes 152 --classes 152 --M 82 --epsion 0.1 --pre_step 600 --temperature 1 --start 0  --reg2 0.1 --logalpha 100 --objective $1
# # # wait
# # # python main.py --dataname S_trans_subst_10 --num_classes 152 --classes 152 --M 82 --epsion 0.1 --pre_step 600 --temperature 1 --start 0 --reg2 0.1 --logalpha 100 --objective $1
# # # wait
# # # python main.py --dataname tweet-original-order_trans_subst_10 --num_classes 89 --classes 89 --M 110  --epsion 0.1 --pre_step -1 --temperature 1 --start 0 --reg2 0.1 --logalpha 100 --objective $1