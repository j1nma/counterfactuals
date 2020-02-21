# Causal Inference Analysis with Neural Networks
Master Thesis for the Double Degree Master Program “Software Engineering” at FH Technikum Wien, Austria.

## About
In recent years, causal inference has changed the direction of statisticians by providing mathematical tools when translating assumptions in data to estimable causal parameters. These tools have shown extensive social and biomedical pertinence for studying how much change a variable causes in another. 

This change can be captured by counterfactuals, a widely used form of causal inference in observational studies. Its definition and notation provides methods to measure, for example, the expected increase in an outcome Y (i.e. blood pressure) as a treatment T (i.e. take a certain medication) changes from 0 (“control” or “no medication taken”) to 1 (“treated” or “the medication is taken”). In this way, it is of interest to capture, for instance, what would have happened to a patient’s blood pressure if he/she had taken a certain medication, knowing that this did not occur. 
Counterfactual aspects in machine learning appear in situations where the learning agent cannot know the feedback it would have received for an action it did not perform. This entails the problem of counterfactual prediction: the feature distribution of the test set differs from that of the train set. 

If counterfactuals are not accounted for in the data fed into a neural network, its performance will be poorer since the discrepancy in the distributions between treated and control populations impede the learner from generalizing on the counterfactual domain from the factual one. If almost no men ever received a certain medication, inferring how men would have reacted to it is unreliable. 

This work studies the effect of balanced representation learning as a preprocessing step for neural network libraries through three approaches: (1) measure the accuracy of a convolutional neural network when learning counterfactuals, (2) compare the accuracy with and without a balanced neural network and, finally, (3) if counterfactuals are identified during training, test a neural network with and without these counterfactuals. 

Adapting a problem by enforcing a neural network to learn balanced representations as with the second approach improves the accuracy of counterfactual inference.

---

## Installing

Clone the repository and under the root folder, run:

Note: on Linux, checkout from branch "ubuntu".

*Python 2*
```shell script
pip install -r requirements.txt
```

*Python 3*
```shell script
pip3 install -r requirements.txt
```
---

## Running

Access the working directory (the root folder 'counterfactuals').

```shell script
cd counterfactuals
```

Set Python path under "./counterfactuals":
```shell script
export PYTHONPATH=$PYTHONPATH:`pwd` 
```

### NN4, CNN, NN4 Variational Bayes
Train architecture (nn4, cnn or nn4_vb) with hyperparameters from the config file, i.e. “train_nn4_ihdp.txt”.
```shell script
python3 ./counterfactuals/net_train.py ./counterfactuals/configs/train_nn4_ihdp.txt
```

Architectures:
* nn4: Feedforward, 4 layers
* cnn: Convoluted Neural Network, 1D convolution, average pooling, twice
* nn4_vb: same as nn4 but with Variational Bayes

Note: if running on PyCharm, edit the config file as this:
```shell script
--outdir
results/ihdp/nn4/
```

### Gridsearch CNN

Edit gridsearch hyperparameters at a config file:
* a: architecture (cnn)
* iterations: (300)
* learning_rate: (0.002)
* weight_decay: (0.0001)
* experiments: number of experiments to train on (10)
* learning_rate_factor: (0.99)
* learning_rate_steps: (3000)
* num_workers: number of processes to use for computation (depends on device) (2)
* batch_size_per_unit: (32)
* seed: (1)
* outdir: (results/ihdp/gridsearch/)
* data_dir: (data/)
* data_train: (ihdp_npci_1-100.train.npz)
* data_test: (ihdp_npci_1-100.test.npz

```shell script
python3 ./counterfactuals/cnn_gridsearch.py ./counterfactuals/configs/gridsearch_cnn_ihdp.txt
```

Note: if running on PyCharm, edit the config file as this:
```shell script
--outdir
results/ihdp/gridsearch/
```

### Counterfactual Regression
Train architecture (cfr) with hyperparameters from the config file, i.e. “cfr_param_search_ihdp.txt”.
This will do both balanced (p_alpha = 1) and unbalanced runs (p_alpha = 0).

```shell script
 python3 ./counterfactuals/cfr/cfr_param_search.py ./counterfactuals/configs/cfr_param_search_ihdp.txt 2
```

Note: if running on PyCharm, edit the config file as this:
```shell script
outdir='results/ihdp/cfr_param_search'
data_dir='../data/'
```

---

## Data
The data is located under the directory "data".
It contains a numpy data file of 100 experiments of IHDP dataset as in _Learning Representations for Counterfactual Inference_, Johansson et al. (2016).

---

## Results
The results will be located under the "counterfactuals/results" folder.
