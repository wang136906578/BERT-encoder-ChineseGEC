# Chinese Grammatical Correction Using BERT-based Pre-trainedModel
Code for BERT-encoder introduced in the AACL-IJCNLP2020 paper [Chinese Grammatical Correction Using BERT-based Pre-trainedModel](https://www.aclweb.org/anthology/2020.aacl-main.20/).
If you find this work helpful in your research, please cite as:
```
@inproceedings{wang-etal-2020-chinese-grammatical-correction,
    title = "{C}hinese Grammatical Correction Using {BERT}-based Pre-trained Model",
    author = "Wang, Hongfei  and
      Kurosawa, Michiki  and
      Katsumata, Satoru  and
      Komachi, Mamoru",
    booktitle = "Proceedings of the 1st Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics and the 10th International Joint Conference on Natural Language Processing",
    year = "2020",
    url = "https://www.aclweb.org/anthology/2020.aacl-main.20"
}
```
# Requirements
- python >= 3.5
- PyTorch == 1.1.0
- pytorch_transformers >= 1.2.0
- other depencies required by [fairseq](https://github.com/pytorch/fairseq)
# How to use
## Download the data and the pre-trained model
The official NLPCC2018_GEC training and test data can be downloaded from (https://github.com/zhaoyyoo/NLPCC2018_GEC). In the official training data, one error sentences may have multiple corrections, we divide them into seperate parts. Since there are no official development data, we randomly extract 5,000 sentences from the training data as the development data. We also segment all sentences into characters. Here is our preprocessed data (https://drive.google.com/file/d/1Huy3QHN6hwfXOE_WjnRERBZyHffuRkv6/view?usp=sharing).

For pre-trained model, see (https://github.com/ymcui/Chinese-BERT-wwm). The pre-trained model we used in our work is ``` RoBERTa-wwm-ext, Chinese ```
## Use fairseq command to turn the data into binary datasets
```
python preprocess.py  --user-dir ./user  \
--task bert_translation \
--srcdict $PATH_TO_PRE_TRAINED_MODEL/vocab.txt  \
-s src -t trg \
--bert-name $PATH_TO_PRE_TRAINED_MODEL \
--destdir $DATA_BIN_DIR  \
--trainpref $PATH_TO_DATA/train \
--validpref $PATH_TO_DATA/dev \
--testpref $PATH_TO_DATA/test
```
## Train the model
```
python train.py $DATA_BIN_DIR \
--seed 1 --user-dir ./user/ \
--optimizer adam \
--task bert_translation --batch-size 32 \
--arch bert_nmt --max-epoch 20 \
--save-dir  $SAVED_MODEL \
-s src -t trg --lr 0.00003
```
## Generate
```
python generate.py $DATA_BIN_DIR \
--task bert_translation  -s src -t trg  --user-dir ./user \
--path $SAVED_MODEL/checkpoint_best.pt --batch-size 32 \
--beam 12  > $OUTPUT_PATH/bert_encoder.txt
```
