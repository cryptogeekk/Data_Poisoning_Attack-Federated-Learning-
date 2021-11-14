def get_model(dataset):
    from tensorflow import keras

    if dataset == "mnist":
        print("using mnist model")
        model=keras.models.Sequential([
                keras.layers.Flatten(input_shape=[784,]),
                keras.layers.Dense(256,activation='tanh'),
                keras.layers.Dense(128,activation='tanh'),
                keras.layers.Dense(10,activation='softmax')
            ])
    
    elif dataset == "cifar":
        model=keras.models.Sequential([
            keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3), padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),
    
            keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),

            keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),
    
            keras.layers.Flatten(),
            keras.layers.Dense(1024, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])

    elif dataset == "fashion":
        model=keras.models.Sequential([
            keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'),
            
            keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.2),
        
            keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.2),
    
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(10, activation='softmax')
        ])

    return model


def model_average(client_weights):
    import numpy as np
    average_weight_list=[]
    for index1 in range(len(client_weights[0])):
        layer_weights=[]
        for index2 in range(len(client_weights)):
            weights=client_weights[index2][index1]
            layer_weights.append(weights)
        average_weight=np.mean(np.array([x for x in layer_weights]), axis=0)
        average_weight_list.append(average_weight)
    return average_weight_list


def change_label(y, l1, l2):
    import numpy as np
    from keras.utils.np_utils import to_categorical

    y_new = np.copy(y)
    flag = 0
    for i in range(len(y)):
        if y[i] == l1:
            if flag != 0:
                y_new[i] = l2
            else:
                flag = 1
    y_new = to_categorical(y_new)

    return y_new