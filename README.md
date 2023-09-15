# ALGM based on GDMRFF

Code & data accompanying the paper ["A Generalized Deep Markov Random Fields Framework for Fake News Detection"].

### How to use our proposed framework(GDMRFF)

To explicitly captures shared fake news properties based on the entire datatset, we propose a graph-theoretic framework, called Generalized Deep Markov Random Fields Framework (GDMRFF), that inherits the capability of deep learning while at the same time exploiting the correlations among the news articles 

You can replace `multi_detection.py` in the `codes` folder with any other DNN-based fake news detection models to extract high-level features, allowing our GDMRFF to improve their performance.


### Prerequisites

This code is written in python 3.7. 
You will need to install a few python packages in order to run the code.
The needed packages are listed as follows:

- [ ] numpy                     1.21.6 
- [ ] scipy                     1.7.3 
- [ ] scikit-learn              1.0.2 
- [ ] torch                     1.12.1 
- [ ] torchvision               0.13.1 
- [ ] tqdm                      4.64.0 
- [ ] transformers              3.4.0 



* Notes: 
    - The processd Twitter dataset can be found at: https://pan.baidu.com/s/1uRgKv2GipCYwpCMTpYc2QA  ( extraction code: 8dqr )
    - Due to copyright issues, the processed Weibo dataset cannot be released directly. Please refer to the original paper that released the dataset.
  
  
### Acknowledgment

Our implementation is mainly based on follows. Thanks for their authors.
https://github.com/cyxanna/CAFE
https://github.com/DeepGraphLearning/GMNN

    


