# Translation with pytorch lighting

# Requirements and Installation
Python version = 3.7
Pytorch = 1.13.1+cu117

`pip install -r requirements.txt`

# Getting Started
```python
parser.add_argument('-b', '--batch', default=32, type=int,
					help='number of each process batch number')
parser.add_argument('-n', '--mname', default="facebook/mbart-large-cc25", type=str,
					help='model name in huggingface')
parser.add_argument('-d', '--dataset', default="wmt14", type=str,
                    help='dataset name in huggingface')
```

Train and test with `python main.py -b 32 -n facebook/mbart-large-cc25 -d wmt14`
