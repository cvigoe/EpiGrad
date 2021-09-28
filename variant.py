variant = dict(
    mlflow_uri="http://128.2.210.74:8080",
    ID_model_name= 'cifar10',       # 'cifar10', 'mnist', 'svhn'
    OOD_model_name= 'mnist',        # 'cifar10', 'mnist', 'svhn'
    k= None,
    lr= 1e-4,
    lmbda= None,
    epochs= 5,
    batch_size= 64,
    batch_size_epi= 1,
    num_tests= 1000,
    max_ent= False,
    num_ROC= 20,
    MC_iters = 1000,
)   