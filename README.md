# Test-Time Gradient Based OOD

To run a set of experiments, specify the datasets, number of tests, and whether to use deep variants in `variant.py`. This code expects to use an MLFlow remote server, whose uri is also specified in `variant.py`. Specify the methods to compare in `main.py` using `epistemic_test_functions`, and then run

```
python -m pudb main.py "MLFlow_Experiment_Name" "MLFlow_run_prefix" "MLFlow note"
```
