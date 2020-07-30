# MisinformationDetection
Combining Temporal Graph Attention Networks and Hawkes Processes to model the dissemination of misinformation via social media as a time-series event propagation.

We use the twitter15/16 datasets (Ma et. al) as is the standard for assessing the performance of many state of the art misinformation detection models. These along with the extra crawled twitter features can be found in the rumor_detection_acl2017 directory here: https://drive.google.com/drive/u/0/folders/1x1aywVkcoArlKiZjLN_If_YcspAI39Np

Data preparation is performed within the files utils.py, text_preprocessing.py and dataset.py. These codes were largely built by the authors of 'Fake News Detection Using Machine Learning on Graphs'. We extend the script dataset.py to build a dynamic graph to illustrate the misinformation propagation over time.

'encoder_decoder.py' combines the encoder and decoder. This architecture is explained here: https://nlp.seas.harvard.edu/2018/04/03/attention.html. A temporal graph sum constructs the encoder, creating a set of node embeddings. This method is from the paper 'Temporal Graph Networks for Deep Learning on Dynamic Graphs'. The veracity prediction task requires performing graph learning on the final state of the dynamic graph to make a prediction on the veracity of the source claim. This method is from the paper 'GCAN: Graph Aware Co-Attention Networks for Explainable Fake News Detection on Social Media'.

'train.py' is the training script for our model.

The files 'Week # Model Implementation.ipynb' provide code breakdowns for my supervisory research partner Qiang Zhang (UCL Centre for Artificial Intelligence).
