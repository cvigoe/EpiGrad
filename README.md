# Test-Time Gradient Based OOD

To run a set of experiments, specify the datasets, number of tests, and whether to use deep variants in `variant.py`. This code expects to use an MLFlow remote server, whose uri is also specified in `variant.py`. Specify the methods to compare in `main.py` using `epistemic_test_functions`, and then run

```
python -m pudb main.py "MLFlow_Experiment_Name" "MLFlow_run_prefix" "MLFlow note"
```
Code is organised as follows:
```
.
├── .gitignore                   
├── README.md                   
├── epistemic_tests.py                      # Where all the gradient-based OOD methods are defined
├── helpers.py                              # Helper functions for plotting, AUC calculation etc.
├── main.py                                 # Main script to run gradient-based OOD experiments
├── main_ortho.py                           # Main script to run Orthonormal Certs experiments
├── orthonormal_certs.py                    # Where Orthonormal Certs are defined
└── roc.py                                  # Functions to caluclate ROC
```    
Hyperparameters should be specified in a file called `variant.py` in the top directory, and its contents should look like:
```
variant = dict(
    mlflow_uri="http://128.2.210.74:8080",
    num_tests= 10000,                             # Number of ID or OOD images to use when testing each method
    deep=False,                                   # Whether to use entire network or just last layer
    model_names=['mnist', 'svhn', 'cifar10'],     # Which datasets & pretrained networks to use
    )
```    
