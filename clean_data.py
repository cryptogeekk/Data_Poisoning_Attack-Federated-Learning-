def get_clean_data(dataset):
    import numpy as np
    
    from sklearn.model_selection import train_test_split
    from keras.utils.np_utils import to_categorical
    
    if dataset == "mnist":
        from keras.datasets import mnist
        (x_train, y_train), (x_te, y_te) = mnist.load_data()
        x_train=np.reshape(x_train,(x_train.shape[0],-1))
        x_te=np.reshape(x_te,(x_te.shape[0],-1))
    
    if dataset == "cifar":
        from keras.datasets import cifar10
        (x_train, y_train), (x_te, y_te) = cifar10.load_data()

    if dataset == "fashion":
        from tensorflow.keras.datasets import fashion_mnist
        (x_train, y_train), (x_te, y_te) = fashion_mnist.load_data()
        x_train = x_train.reshape((60000,28,28,1))
        x_te = x_te.reshape((10000,28,28,1))

    x = np.array(x_train)
    y = np.array(y_train)

    xt = np.array(x_te)
    yt = np.array(y_te)
    
    x = x / 255.0
    xt = xt / 255.0

    y = to_categorical(y)
    yt = to_categorical(yt)

    return x, y, xt, yt





