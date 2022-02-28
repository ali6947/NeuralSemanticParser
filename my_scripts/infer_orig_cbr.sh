# ARCHIVE_DIR="/mnt/infonas/data/alirehan/semantic_parsing/smbop/try_train/experiments/zippy-apricot-bear_rat_layers4_batch_size24" # trained on randomly selected same schema cases
# ARCHIVE_DIR="/mnt/infonas/data/alirehan/smbop/try_train/experiments/hilly-corn-baboon_rat_layers4" # trained on gold cases for spider vanilla
# ARCHIVE_DIR="/mnt/infonas/data/alirehan/semantic_parsing/smbop/try_train/experiments/jumpy-crimson-toucan_batch_size3" # trained on gold cases. Cases used in utt_aug also
# ARCHIVE_DIR="/mnt/infonas/data/alirehan/semantic_parsing/smbop/try_train/experiments/bluesy-cardinal-barracuda_batch_size3" # trained on mix of gold cases and soft ret (w span,leaf loss) cases. Cases used in utt_aug also
# ARCHIVE_DIR="/mnt/infonas/data/alirehan/semantic_parsing/smbop/try_train/experiments/silly-gamboge-akbash_batch_size4_grad_acum3" # trained on top 5 same schema cases, Cases used in utt_aug also
ARCHIVE_DIR="/mnt/infonas/data/alirehan/semantic_parsing/smbop/try_train/experiments/pretty-saffron-catfish_grad_acum1" # vanilla mix top 5 cases same schema
DBID="car_1"
RESULTS="results"


##### VANILLA SPIDER GOLD CASES

# python3 eval_orig_cbr.py \
#   --archive_dir $ARCHIVE_DIR \
#   --dev_path "/mnt/infonas/data/alirehan/semantic_parsing/pickle_objs/bigbird_base/complete_spider_val_with_gold_cases_bigbird_base.pkl" \
#   --output $ARCHIVE_DIR/op_cases1.sql \
#   --output_gold $ARCHIVE_DIR/gold_op_cases1.sql \
#   --db_id $DBID

# python3 smbop/eval_final/evaluation.py \
#   --gold $ARCHIVE_DIR/gold_op_cases1.sql \
#   --pred $ARCHIVE_DIR/op_cases1.sql \
#   --etype match \
#   --db  dataset/database  \
#   --table dataset/tables.json \
#   --fout $RESULTS/tmpout.txt > /dev/null

# cat $RESULTS/tmpout.txt

##### VANILLA SPIDER

# python3 eval_orig_cbr.py \
#   --archive_dir $ARCHIVE_DIR \
#   --dev_path "/mnt/infonas/data/alirehan/semantic_parsing/pickle_objs/bigbird_base/all_InstanceObj1031_spider_val_bigbird_base_w_samedb_cases.pkl" \
#   --output $ARCHIVE_DIR/op_cases1.sql \
#   --output_gold $ARCHIVE_DIR/gold_op_cases1.sql \
#   --db_id $DBID

# python3 smbop/eval_final/evaluation.py \
#   --gold $ARCHIVE_DIR/gold_op_cases1.sql \
#   --pred $ARCHIVE_DIR/op_cases1.sql \
#   --etype match \
#   --db  dataset/database  \
#   --table dataset/tables.json \
#   --fout $RESULTS/tmpout.txt > /dev/null

# cat $RESULTS/tmpout.txt


##### SYNON SPIDER

# python3 eval_kaggle_orig_cbr.py \
#   --archive_dir $ARCHIVE_DIR \
#   --dev_path "/mnt/infonas/data/alirehan/semantic_parsing/pickle_objs/bigbird_base/synon_spider_val_all_inst_concat_cases.pkl" \
#   --output $ARCHIVE_DIR/op_cases1.sql \
#   --output_gold $ARCHIVE_DIR/gold_op_cases1.sql \

# python3 smbop/eval_final/evaluation.py \
#   --gold $ARCHIVE_DIR/gold_op_cases1.sql \
#   --pred $ARCHIVE_DIR/op_cases1.sql \
#   --etype match \
#   --db  dataset/database  \
#   --table /mnt/infonas/data/awasthi/semantic_parsing/smbop/dataset/noisy_schema/seed_0/tables.json \
#   --fout $RESULTS/tmpout.txt > /dev/null

# cat $RESULTS/tmpout.txt

##### SYNON SPIDER

# python3 eval_kaggle_orig_cbr.py \
#   --archive_dir $ARCHIVE_DIR \
#   --dev_path "/mnt/infonas/data/alirehan/semantic_parsing/pickle_objs/bigbird_base/canon_spider_val_all_inst_concat_cases.pkl" \
#   --output $ARCHIVE_DIR/op_cases1.sql \
#   --output_gold $ARCHIVE_DIR/gold_op_cases1.sql \

# python3 smbop/eval_final/evaluation.py \
#   --gold $ARCHIVE_DIR/gold_op_cases1.sql \
#   --pred $ARCHIVE_DIR/op_cases1.sql \
#   --etype match \
#   --db  dataset/database  \
#   --table /mnt/infonas/data/awasthi/semantic_parsing/smbop/dataset/noisy_schema/canonicalized_tables.json \
#   --fout $RESULTS/tmpout.txt > /dev/null

# cat $RESULTS/tmpout.txt

######KAGGLE 

# python3 eval_kaggle_orig_cbr.py \
#   --archive_dir $ARCHIVE_DIR \
#   --dev_path "/mnt/infonas/data/alirehan/semantic_parsing/smbop/kaggle-dbqa/pickles_bigbird/all_kaggle_inst_concat_cases.pkl" \
#   --output $ARCHIVE_DIR/op_cases1.sql \
#   --output_gold $ARCHIVE_DIR/gold_op_cases1.sql \

# python3 smbop/eval_final/evaluation.py \
#   --gold $ARCHIVE_DIR/gold_op_cases1.sql \
#   --pred $ARCHIVE_DIR/op_cases1.sql \
#   --etype match \
#   --db  smbop/kaggle-dbqa/databases  \
#   --table smbop/kaggle-dbqa/tables/all_tables.json \
#   --fout $RESULTS/tmpout.txt > /dev/null

# cat $RESULTS/tmpout.txt


##### RETRIEVED CASES SPIDER

# python3 eval_orig_cbr.py \
#   --archive_dir $ARCHIVE_DIR \
#   --dev_path "/mnt/infonas/data/alirehan/semantic_parsing/pickle_objs/bigbird_base/spider_val_same_schema_high_cossim_cases.pkl" \
#   --output $ARCHIVE_DIR/op_cases1.sql \
#   --output_gold $ARCHIVE_DIR/gold_op_cases1.sql \
#   --db_id $DBID

# python3 smbop/eval_final/evaluation.py \
#   --gold $ARCHIVE_DIR/gold_op_cases1.sql \
#   --pred $ARCHIVE_DIR/op_cases1.sql \
#   --etype match \
#   --db  dataset/database  \
#   --table dataset/tables.json \
#   --fout $RESULTS/tmpout.txt > /dev/null

# cat $RESULTS/tmpout.txt


######## EVAL SPLITS RANDOM CASES
# echo "here"
# python3 eval_orig_cbr_splits.py \
#   --archive_dir $ARCHIVE_DIR \

# python3 smbop/eval_final/evaluation.py \
#   --gold $RESULTS/bigbird_splits/split_2_car_1_gold \
#   --pred $RESULTS/bigbird_splits/split_2_car_1_pred \
#   --etype all \
#   --db  dataset/database  \
#   --table dataset/tables.json \
#   --fout $RESULTS/tmpout.txt > /dev/null

# cat $RESULTS/tmpout.txt


######## EVAL SPLITS TOP 5 SAME SCHEMA CASEs FROM TRAIN FILE 

# python3 eval_orig_cbr_splits_cases_in_same_pkl.py \
#   --archive_dir $ARCHIVE_DIR \


########### EVAL RET VANILLA SPIDER WITH BEAM FILES

python3 -u eval_kaggledb_beam_orig_cbr.py \
        --archive_dir=$ARCHIVE_DIR \
        --dev_path=/mnt/infonas/data/alirehan/semantic_parsing/pickle_objs/bigbird_base/spider_val_bigbird_ret_soft_cases_no_leafs_val_double_check.pkl \
        --table_path=dataset/tables.json \
        --dataset_path=dataset/database \
        --output=$ARCHIVE_DIR/op_cases1.sql \
        --output_gold=$ARCHIVE_DIR/gold_op_cases1.sql
        


python3 -u smbop/eval_final/evaluation.py \
  --gold $ARCHIVE_DIR/gold_op_cases1.sql \
  --pred $ARCHIVE_DIR/op_cases1.sql \
  --etype all \
  --db  dataset/database  \
  --table dataset/tables.json \
  --fout $RESULTS/tmpoutuw.txt > /dev/null

cat $RESULTS/tmpoutuw.txt
