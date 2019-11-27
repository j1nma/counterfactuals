# Causal Inference as a Preprocessing Step for Neural Network Libraries

## Running

Train architecture (nn4 or cnn) with hyperparameters from the config file.
```shell script
python3 ./counterfactuals/net_train.py ./configs/train_nn4_ihdp.txt
```

Test nn4 architecture with hyperparameters from the config file.
```shell script
python3 ./counterfactuals/net_test.py ./configs/test_nn4_ihdp.txt
```

Edit gridsearch hyperparameters (steps 1 to 7), the rest must be at the config file.
```shell script
python3 ./counterfactuals/gridsearch.py ./configs/gridsearch_cnn_ihdp.txt
```