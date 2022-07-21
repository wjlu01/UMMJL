# UMMJL
Unified Multi-modal Multi-task Joint Learning for  Language-vision Relation Inference

By Wenjie Lu and Dong Zhang.

## Environment
- python=3.7
- torch=1.7.1
- numpy=1.18.3
- pytorch-pretrained-bert=0.6.2

## Dataset
Download [Bloomberg's images](https://drive.google.com/file/d/1KkeMBo32gDEfrbmUjTEX4whpdda_wrYn/view?usp=sharing), [Flicker 30k's images](https://drive.google.com/file/d/1C5wvXzZfB2u05LyKj9v2_rPipiWheQ6_/view?usp=sharing), [MVSA's images](https://drive.google.com/file/d/1bgrvWzhKCBABNSHgu2j8BAxrx3jyTkre/view?usp=sharing) and [GossipCop's images](https://drive.google.com/file/d/1H7Am9U_bgwaBrKHCTqHwhjm0HqYCu8IH/view?usp=sharing).

We expect the directory structure to be the following:
```
datasets/
├── bloomberg/
│   ├── rel_img/
│   ├── train
│   ├── dev
│   └── test
├── caption/
│   ├── img/
│   ├── train
│   └── test
└── fake_news/
│   ├── img/
│   ├── train
│   └── test
└── senti/
    ├── img/
    ├── train
    └── test
```

## Pre-trained Models
```
pretrained/
├── embeddings/
│   ├──en-fasttext-crawl-300d-1M
│   ├──common_characters_large
├── bert-base-uncased/
│   ├── vocab.txt
│   ├── bert_config.json
│   └── pytorch_model.bin
└── resnet/
    └──resnet152-b121ed2d.pth
```
## Model Training
Train, evaluate, and test the three sub-tasks in Bloomberg with the three auxiliary tasks respectively.
```
sh scripts/train.sh
```

Example command:
```
python main.py --cuda 1 \
--mode 0 \
--auxiliary_task_split_file datasets/caption/ \
--auxiliary_task_img_dir datasets/caption/img/ \
--task_name caption \
--model_dir save_model/caption_mask_img1_1e-5_1e-5/ \
--class_name img --mask_part img_1 \
--dropout 0.5 --batch_size 16 \
--lr_1 1e-5 --lr_2 1e-5
```
Description of some parameters in main.py:
- class_name: name for the main task: img, txt, it.
- task_name: name for the auxiliary task: senti, caption, fake_news.
- mode: learning phase: 0 for training and 1 for testing.

## Model Testing
### Download Checkpoints
We provide links to download our checkpoints in `save_model/README.md`.

### Testing
Test the three sub-tasks in Bloomberg with the three auxiliary tasks respectively.
```
sh scripts/test.sh
```

## Acknowledgement
The code is based on the [RpBERT](https://github.com/Multimodal-NER/RpBERT). Thanks for the great work.

