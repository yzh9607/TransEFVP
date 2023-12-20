# TransEFVP
Studying the effect of single amino acid variations (SAVs) on protein structure and function is integral to advancing the understanding of molecular processes, evolutionary biology, and disease mechanisms. Screening for deleterious variants is one of the crucial issues in precision medicine. Here, we propose a novel computational approach, TransEFVP, based on large-scale protein language model embeddings and a transformer-based neural network to predict disease-associated SAVs. The model adopts a two-stage architecture: the first stage is designed to fuse different feature embeddings through a transformer encoder. In the second stage, a support vector machine (SVM) model is employed to quantify the pathogenicity of SAVS after dimensionality reduction. The prediction performance of TransEFVP on blind test data achieves a Matthews correlation coefficient of 0.751, an F1-score of 0.846, and an area under the receiver operating characteristic curve of 0.871, higher than the existing state-of-the-art methods. The benchmarking results demonstrate that TransEFVP can be explored as an accurate and effective SAV pathogenicity prediction method.
# Requirements
pandas==1.5.3  numpy==1.23.3
scikit-learn==1.0.2
tensorboard==2.8.0
tensorflow-gpu==2.4.1
pytorch==1.12.1
