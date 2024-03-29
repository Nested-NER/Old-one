
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

Before running the code, please put word embeddings to the path "/model/word2vec/".

### Data format

R7 - 57 reporter cells , on the other hand , signaled induced activity of the lytic origin of EBV replication ( ori Lyt ) .

NN NN NN NN NNS , IN DT JJ NN , VBD VBN NN IN DT JJ NN IN NN NN ( NN NN ) .

0,5 G#cell_line|16,21 G#DNA|22,24 G#DNA

The first line is a sentence. The second line is POS tags. The third line is the location (start,end] and type of entity separated by "|". For example, "0,5 G#cell_line" denotes "R7 - 57 reporter cells"  is a "cell_line".


### Configuration
All configuration are listed in config.py. Please verify parameters before running the codes.
>- Please download BERT-Large-Uncased (https://github.com/google-research/bert), copy it to the path "./bert_model/large/"  and train the  model in DTE+BERT  mode.

### Usage
#### Training
>- python process_data.py
>- python train.py 

If you run DTE for contextual network, please set  self.use_bert = False, self.if_DTE = True in config.py.

If you run BERT for contextual network, please set  self.use_bert = True, self.if_DTE = False in config.py. 

If you run DTE+BERT for contextual network, please set  self.use_bert = True, self.if_DTE = True in config.py.

#### Testing
>- python test.py

### Test Best Model:
The best model is located on "./model" path. You can change the "test_model_path" to choose model and run
"python test.py" to evaluate it.
