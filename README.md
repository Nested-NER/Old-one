# Toi-SCNN-DTE:Text-of-Interest detection with Stacked Convolutional Neural Network and Dual Transformer Encoders

### Environment
#### Python packages
>- python==3.6
>- pytorch==1.0.0
>- gensim==3.3.0
>- numpy==1.15.0
>- pandas==0.20.3

**(pip install -r requirements.txt)**
### word embeddings download links
>- ACE04/05 : [glove 100d](https://drive.google.com/open?id=1qDmFF0bUKHt5GpANj7jCUmDXgq50QJKw)
>- GENIA : [wikipedia-pubmed-and-PMC-w2v 200d](http://evexdb.org/pmresources/vec-space-models/wikipedia-pubmed-and-PMC-w2v.bin)

Before running the codes, please move word embeddings to "/model/word2vec/".

### Configuration
All configuration are listed in config.py. Please verify parameters before running the codes.

### Usage
#### Training
>- python process_data.py
>- python train.py 

#### Testing
>- python test.py

### Test Best Model:
The best model is located on "./model" path. You can change the "test_model_path" to choose model and run
"python test.py" to evaluate it.
