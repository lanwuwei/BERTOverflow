<p align="center">
  <img src="bertoverflow.png" width="400">
  <br />
</p>


# BERTOverflow
This repository contains pre-trained BERT on StackOverflow data, which has shown state-of-the-art performance (with CRF layer) on software domain NER. The checkpoints can be downloaded [here](https://drive.google.com/drive/folders/1z4zXexpYU10QNlpcSA_UPfMb2V34zHHO?usp=sharing).

For further details, see the accompanying paper:
[Code and Named Entity Recognition in StackOverflow](https://arxiv.org/pdf/2005.01634.pdf)

# Data
We extract 152M sentences from StackOverflow questions and answers.

# Vocabulary
We create 80K cased [WordPiece](https://github.com/huggingface/tokenizers) vocabulary with 2K different UNK symbols:
```
import tokenizers
bwpt = tokenizers.BertWordPieceTokenizer(
    vocab_file=None,
    add_special_tokens=True,
    unk_token='[UNK]',
    sep_token='[SEP]',
    cls_token='[CLS]',
    clean_text=True,
    lowercase=False,
    handle_chinese_chars=True,
    strip_accents=True,
    wordpieces_prefix='##'
)
bwpt.train(
    files=["all_lines_from_ques_ans_xml_excluded_ann.txt"],
    vocab_size=80000,
    min_frequency=30,
    limit_alphabet=2000,
    special_tokens=['[PAD]', '[UNK]', '[CLS]', '[MASK]', '[SEP]']
)
bwpt.save("./","soft-bert-vocab")
```

# TF Records
We split large file and create TF Records parallelly:
```
split -l 400000 ../../data/my_data.all my-data- --verbose

ls ../saved_model/softbert/raw_txt_data/ | xargs -n 1 -P 16 -I{} python create_pretraining_data.py --input_file=../saved_model/softbert/raw_txt_data/{} --output_file=../saved_model/softbert/tf_records_data/{}.tfrecord --vocab_file=../saved_model/softbert/vocab.txt --max_seq_length=128 --max_predictions_per_seq=20 --masked_lm_prob=0.15 --random_seed=12345 --dupe_factor=5 --do_whole_word_mask=False --do_lower_case=False
```

# Pre-training
The pre-training is conducted on TPU v2-8 with Tensorflow 1.15:
```
python3 run_pretraining.py --input_file=gs://softbert_data/processed_data/*.tfrecord --output_dir=gs://softbert_data/model_base/ --do_train=True --do_eval=True --bert_config_file=gs://softbert_data/model_base/bert_config.json --train_batch_size=512 --max_seq_length=128 --max_predictions_per_seq=20 --num_train_steps=1500000 --num_warmup_steps=10000 --learning_rate=1e-4 --use_tpu=True --tpu_name=$TPU_NAME --save_checkpoints_steps 100000
```

