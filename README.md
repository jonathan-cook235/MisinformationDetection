# MisinformationDetection
Combining Temporal Graph Attention Networks and Hawkes Processes to model the dissemination of misinformation via social media as a time-series event propagation.

We use the twitter15/16 datasets (Ma et. al) as is the standard for assessing the performance of many state of the art misinformation detection models.

Within the 'data' directory is the 'rumour_detection_acl2017' directory, which contains the datasets. Data preparation is performed within the files utils.py, text_preprocessing.py and dataset.py. These codes were built by the authors of 'Fake News Detection Using Machine Learning on Graphs'.

The file 'graphs.py' constructs the dynamic graph that encapsulates the temporal and geometric propagation of misinformation via social media interactions. It is an extension of the codes also written by the authors of 'Fake News Detection Using Machine Learning on Graphs'.

'temporal_graph_sum.py' constructs the encoder, creating a set of node embeddings. This method is from the paper 'Temporal Graph Networks for Deep Learning on Dynamic Graphs'.

'veracity_prediction.py' performs graph learning on the final state of the dynamic graph to make a prediction on the veracity of the source claim. This method is from the paper 'GCAN: Graph Aware Co-Attention Networks for Explainable Fake News Detection on Social Media'.

The files 'Week # Model Implementation' provide code breakdowns for my supervisory research partner Qiang Zhang (UCL Centre for Artificial Intelligence).
