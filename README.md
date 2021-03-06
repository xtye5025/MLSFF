# MLSFF

This repository contains the reference code for the paper Multi-Label Semantic Feature Fusion for Remote Sensing Image Captioning. Complete data and code will be available soon.

## Dependencies
This is the list of python requirements:
- [pytorch==1.6.0](http://pytorch.org/)
- [torchvision==0.7.0](https://github.com/pytorch/vision)
- [torchtext==0.6.0](https://github.com/pytorch/text)
- [spaCy==3.0.7](https://github.com/explosion/spaCy)
- [coco-caption](https://github.com/jiasenlu/coco-caption)
- python==3.7.6
- numpy==1.21.1
- pandas==1.3.1
- Pillow==8.1.2
- h5py==3.4.0
- tqdm==4.56.0
- scikit-learn==0.24.1
- scipy==1.6.3
- tensorboard==1.15.0
- tensorboardX==2.0
- en-core-web-lg==3.0.0

Create the conda environment with the provided ```environment.yml``` file:
    ```
    conda env create -f dependencies.yml
    ```  

## Training and Evaluation
### Data Preparation

### Pretrained Model

### Training
```
python train.py --batch_size 10 --use_num_attr 2 --checkpoint_dir ./stored_weights/ --num_epoches 50
```
### Evaluation

## Visualization

## More Information and Experimental Results

![word counts ucm](Figures/word_count.png)
