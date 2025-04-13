# ECE1508

## Adversarial Training of ResNets for Enhanced Robustness

`baseModel.py` and `adversarialModel.py` contain the entrypoints for this code. Edit the params dictionary (i.e. the keys "mode" and "dataset") to either determine whether to train/test the model and which dataset to use base/combined/adversarial when doing so.

`singleSampleTest.py` and `multiSampleTest.py` also touch the entrypoints but only for evaluating all models on some given data.