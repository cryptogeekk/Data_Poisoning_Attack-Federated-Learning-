def targetted_poison(x, y):
    import numpy as np
    from keras.utils.np_utils import to_categorical

    target_dict = {
        0: 5,
        1: 8,
        2: 9,
        3: 7,
        4: 1,
        5: 0,
        6: 4,
        7: 6,
        8: 2,
        9: 3
    }

    yt=[]

    y_f = np.argmax(y, axis=1)
    
    for i in range(len(x)):
        yt.append(target_dict[y_f[i]])

    yt = np.array(yt)
    yt = to_categorical(yt)

    return x, yt


def diversion_poison(wb, x, y, dataset):
    import numpy as np
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.losses import categorical_crossentropy
    
    from utils import get_model  
    
    model=get_model(dataset)
    
    def inverted_cat_crossentropy(y_true, y_pred):
        ltrue = keras.backend.argmax(y_true)
        lpred = keras.backend.argmax(y_pred)
        is_same = ltrue == lpred
        # loss = 1 - (y_true - y_pred)
        # loss = 2 - categorical_crossentropy(y_true, y_pred)
        # loss2 = categorical_crossentropy(y_true, y_pred) 
        # loss2 = (loss2) / 200
        # sm_loss = 0.005
        # is_small = loss2 < sm_loss
        loss = categorical_crossentropy(y_true, y_pred)
        loss = tf.where(loss==0, 0.001, loss)
        loss = 1 / loss
        return loss
        # return tf.where(is_same, loss, loss2)

    model.compile(loss=inverted_cat_crossentropy,optimizer="adam",metrics=['accuracy'])
    model.set_weights(wb)

    history = model.fit(x, y, epochs= 20, batch_size=32)
    
    y_new = model.predict(x)
    
    y_f = np.argmax(y, axis=1)
    y_new_f = np.argmax(y_new, axis=1)

    diff_samples = []
    for i in range(10):
        diff_samples.append(np.zeros(10))
    for i in range(len(y_f)):
        diff_samples[y_f[i]][y_new_f[i]] = diff_samples[y_f[i]][y_new_f[i]] + 1

    mappings = np.zeros(10)
    for i in range(10):
        mappings[i] = np.argmax(diff_samples[i])  
    print("Mappings:", mappings)

    return x, y_new


def brute_force_poison(x_data, y_data, X_valid, y_valid, dataset):
    import numpy as np
    from keras.utils.np_utils import to_categorical
    from utils import get_model, change_label
    print("Brute force label flipping")
    y_data_f = np.argmax(y_data, axis=1)
    y_vals = np.unique(y_data_f)
    mapping_dict = {}
    for label1 in y_vals:
        acc = 100
        lab = label1
        for label2 in y_vals:
            if label1 != label2:
                print("Flipping ", label1, " to ", label2)
                y_new = change_label(y_data_f, label1, label2)
                model = get_model(dataset)
                model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
                history = model.fit(x_data, y_new, epochs=10)
                result = model.evaluate(X_valid, y_valid)
                accuracy = result[1]
                if accuracy < acc:
                    acc = accuracy
                    lab = label2
        mapping_dict[label1] = lab
    print(mapping_dict)
    y_pois = []
    for i in range(len(y_data_f)):
        y_pois.append(mapping_dict[y_data_f[i]])

    y_pois = np.array(y_pois)
    y_pois = to_categorical(y_pois)

    return x_data, y_pois