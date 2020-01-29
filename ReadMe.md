This project is for referring expression comprehension via semantic-aware network.

**Requirements**

Python 2.7

PyTorch 0.4.1

**Dataset**

Please go to this [site](https://github.com/lichengunc/refer) and follow the instructions to complete installation.

**Feature Extraction**

The pretraine model [BERT](https://github.com/codertimo/BERT-pytorch) is required to generate representations for referring expressions.

**Train**

python tools/train.py --dataset refcoco --splitBy unc

**Test**

python tools/eval.py --dataset refcoco --splitBy unc
