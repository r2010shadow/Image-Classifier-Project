#!/bin/bash

# run train.py 
python train.py --arch vgg16 --gpu True --epochs 10 --hidden_units 100 --learning_rate 0.001
# but cant find floder in the online workspace


# run predict.py
python predict.py 'flowers/test/1/image_06752.jpg' --gpu True --category_names 'cat_to_name.json' --top_k 5
# also dont found flower files in the workspace
