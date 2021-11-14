import numpy as np
import pandas as pd
import matplotlib as plt
from tensorflow import keras

import clean_data
from dataset_divider import divide_without_label
from utils import get_model, model_average
from poisoning import targetted_poison, diversion_poison, brute_force_poison
from client import Client

NUM_CLASSES = 10
MAL_CLIENTS = 5
EPOCHS_PER_CLIENT = 5
COMMUNICATION_ROUNDS = 100
dataset = "fashion"


X_train, y_train, X_valid,y_valid = clean_data.get_clean_data(dataset)
x_data, y_data = divide_without_label(10, X_train, y_train)

x_data_train = np.copy(x_data)
y_data_train = np.copy(y_data)

x_data_tg100 = np.copy(x_data)
y_data_tg100 = np.copy(y_data)

x_data_bf = np.copy(x_data)
y_data_bf = np.copy(y_data)


def create_model():
    model=get_model(dataset)
    weight=model.get_weights()
    return weight
    
    

def train_server(training_rounds, epoch, batch, learning_rate):

    accuracy_list=[]
    client_weight_for_sending=[]

    accuracy_list_gd=[]
    client_weight_for_sending_gd=[]
    
    accuracy_list_t100=[]
    client_weight_for_sending_t100=[]

    accuracy_list_bf=[]
    client_weight_for_sending_bf=[]

    #targetted label flipping poisoning
    if MAL_CLIENTS > 2:
        x_data_tg100[5], y_data_tg100[5] = targetted_poison(x_data_tg100[5], y_data_tg100[5])
        x_data_tg100[2], y_data_tg100[2] = targetted_poison(x_data_tg100[2], y_data_tg100[2])
        x_data_tg100[8], y_data_tg100[8] = targetted_poison(x_data_tg100[8], y_data_tg100[8])

    if MAL_CLIENTS > 1:
        x_data_tg100[4], y_data_tg100[4] = targetted_poison(x_data_tg100[4], y_data_tg100[4])

    
    if MAL_CLIENTS > 0:
        x_data_tg100[6], y_data_tg100[6] = targetted_poison(x_data_tg100[6], y_data_tg100[6])


    #bruteforce posioning
    if MAL_CLIENTS == 5:
        x_to_calc_bf = np.copy(x_data[4])
        x_to_calc_bf = np.append(x_to_calc_bf, x_data[6], axis = 0)
        x_to_calc_bf = np.append(x_to_calc_bf, x_data[5], axis = 0)
        x_to_calc_bf = np.append(x_to_calc_bf, x_data[2], axis = 0)
        x_to_calc_bf = np.append(x_to_calc_bf, x_data[8], axis = 0)

        y_to_calc_bf = np.copy(y_data[4])
        y_to_calc_bf = np.append(y_to_calc_bf, y_data[6], axis = 0)
        y_to_calc_bf = np.append(y_to_calc_bf, y_data[5], axis = 0)
        y_to_calc_bf = np.append(y_to_calc_bf, y_data[2], axis = 0)
        y_to_calc_bf = np.append(y_to_calc_bf, y_data[8], axis = 0)

        x_data_calc_bf, y_data_calc_bf = brute_force_poison(x_to_calc_bf, y_to_calc_bf, X_valid, y_valid, dataset)
        x_poisoned_bf, y_poisoned_bf = divide_without_label(5, x_data_calc_bf, y_data_calc_bf)
        
        x_data_bf[8], y_data_bf[8] = x_poisoned_bf[0], y_poisoned_bf[0]
        x_data_bf[2], y_data_bf[2] = x_poisoned_bf[1], y_poisoned_bf[1]
        x_data_bf[5], y_data_bf[5] = x_poisoned_bf[2], y_poisoned_bf[2]
        x_data_bf[6], y_data_bf[6] = x_poisoned_bf[3], y_poisoned_bf[3]
        x_data_bf[4], y_data_bf[4] = x_poisoned_bf[4], y_poisoned_bf[4]
    
    if MAL_CLIENTS == 2:
        x_to_calc_bf = np.copy(x_data[4])
        x_to_calc_bf = np.append(x_to_calc_bf, x_data[6], axis = 0)
        y_to_calc_bf = np.copy(y_data[4])
        y_to_calc_bf = np.append(y_to_calc_bf, x_data[6], axis = 0)

        x_data_calc_bf, y_data_calc_bf = brute_force_poison(x_to_calc_bf, y_to_calc_bf, X_valid, y_valid, dataset)
        x_poisoned_bf, y_poisoned_bf = divide_without_label(2, x_data_calc_bf, y_data_calc_bf)

        x_data_bf[6], y_data_bf[6] = x_poisoned_bf[0], y_poisoned_bf[0]
        x_data_bf[4], y_data_bf[4] = x_poisoned_bf[1], y_poisoned_bf[1]
        
    if MAL_CLIENTS == 1:
        x_to_calc = np.copy(x_data[4])
        y_to_calc = np.copy(y_data[4])
        x_poisoned_bf, y_poisoned_bf = brute_force_poison(x_to_calc_bf, y_to_calc_bf, X_valid, y_valid, dataset)
        x_data_bf[4], y_data_bf[4] = x_poisoned_bf, y_poisoned_bf


    for index1 in range(1,training_rounds+1):
        
        #gradient diversion poisoning
        if index1 > 10:
            if MAL_CLIENTS == 5:
                x_to_calc = np.copy(x_data[4])
                x_to_calc = np.append(x_to_calc, x_data[6], axis = 0)
                x_to_calc = np.append(x_to_calc, x_data[5], axis = 0)
                x_to_calc = np.append(x_to_calc, x_data[2], axis = 0)
                x_to_calc = np.append(x_to_calc, x_data[8], axis = 0)

                y_to_calc = np.copy(y_data[4])
                y_to_calc = np.append(y_to_calc, y_data[6], axis = 0)
                y_to_calc = np.append(y_to_calc, y_data[5], axis = 0)
                y_to_calc = np.append(y_to_calc, y_data[2], axis = 0)
                y_to_calc = np.append(y_to_calc, y_data[8], axis = 0)

                x_data_calc, y_data_calc = diversion_poison(client_weight_for_sending[index1-2], x_to_calc, y_to_calc, dataset)
                x_poisoned, y_poisoned = divide_without_label(5, x_data_calc, y_data_calc)
                
                x_data_train[8], y_data_train[8] = x_poisoned[0], y_poisoned[0]
                x_data_train[2], y_data_train[2] = x_poisoned[1], y_poisoned[1]
                x_data_train[5], y_data_train[5] = x_poisoned[2], y_poisoned[2]
                x_data_train[6], y_data_train[6] = x_poisoned[3], y_poisoned[3]
                x_data_train[4], y_data_train[4] = x_poisoned[4], y_poisoned[4]
        
            if MAL_CLIENTS == 2:
                x_to_calc = np.copy(x_data[4])
                x_to_calc = np.append(x_to_calc, x_data[6], axis = 0)
                y_to_calc = np.copy(y_data[4])
                y_to_calc = np.append(y_to_calc, y_data[6], axis = 0)

                x_data_calc, y_data_calc = diversion_poison(client_weight_for_sending[index1-2], x_to_calc, y_to_calc, dataset)
                x_poisoned, y_poisoned = divide_without_label(2, x_data_calc, y_data_calc)

                x_data_train[6], y_data_train[6] = x_poisoned[0], y_poisoned[0]
                x_data_train[4], y_data_train[4] = x_poisoned[1], y_poisoned[1]
                
            if MAL_CLIENTS == 1:
                x_to_calc = np.copy(x_data[4])
                y_to_calc = np.copy(y_data[4])
                x_poisoned, y_poisoned = diversion_poison(client_weight_for_sending[index1-2], x_to_calc, y_to_calc, dataset)
                x_data_train[4], y_data_train[4] = x_poisoned, y_poisoned

        print('Training for round ', index1, 'started')

        client_weights_tobe_averaged=[]
        client_weights_tobe_averaged_t100=[]
        client_weights_tobe_averaged_gd=[]
        client_weights_tobe_averaged_bf=[]

        for index in range(len(y_data)):
            print('-------Client-------', index)
            if index1==1:
                print('Sharing Initial Global Model with Random Weight Initialization')
                initial_weight=create_model()
                client=Client(dataset, x_data[index],y_data[index],epoch,learning_rate,initial_weight,batch)
                weight=client.train()
                client_weights_tobe_averaged.append(weight)
                client_weights_tobe_averaged_t100.append(weight)
                client_weights_tobe_averaged_gd.append(weight)
                client_weights_tobe_averaged_bf.append(weight)
            else:
                client=Client(dataset, x_data[index],y_data[index],epoch,learning_rate,client_weight_for_sending[index1-2],batch)
                weight=client.train()
                client_weights_tobe_averaged.append(weight)
                if index1 > 10:
                    #malicious clients training with targetted posioned data
                    client=Client(dataset, x_data_tg100[index],y_data_tg100[index],epoch,learning_rate,client_weight_for_sending[index1-2],batch)
                    weight=client.train()
                    client_weights_tobe_averaged_t100.append(weight)

                    #malicious clients training with gd posioned data
                    client=Client(dataset, x_data_train[index],y_data_train[index],epoch,learning_rate,client_weight_for_sending[index1-2],batch)
                    weight=client.train()
                    client_weights_tobe_averaged_gd.append(weight)

                    #malicious clients training with bf posioned data
                    client=Client(dataset, x_data_bf[index],y_data_bf[index],epoch,learning_rate,client_weight_for_sending[index1-2],batch)
                    weight=client.train()
                    client_weights_tobe_averaged_bf.append(weight)
                
                else:
                    client_weights_tobe_averaged_t100.append(weight)
                    client_weights_tobe_averaged_gd.append(weight)
                    client_weights_tobe_averaged_bf.append(weight)
    

        # benign setting averaging
        client_average_weight=model_average(client_weights_tobe_averaged)
        client_weight_for_sending.append(client_average_weight)
        model=get_model(dataset)
        model.set_weights(client_average_weight)
        model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
        result = model.evaluate(X_valid, y_valid)
        accuracy = result[1]
        print('#######-----Benign acccuracy for round ', index1, 'is ', accuracy, ' ------########')
        accuracy_list.append(accuracy)

        # targetted posioning setting averaging
        client_average_weight_t100=model_average(client_weights_tobe_averaged_t100)
        client_weight_for_sending_t100.append(client_average_weight_t100)
        model = get_model(dataset)
        model.set_weights(client_average_weight_t100)
        model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
        result=model.evaluate(X_valid, y_valid)
        accuracy=result[1]
        print('#######-----Targetted acccuracy for round ', index1, 'is ', accuracy, ' ------########')
        accuracy_list_t100.append(accuracy)

        # gd posioning setting averaging
        client_average_weight_gd=model_average(client_weights_tobe_averaged_gd)
        client_weight_for_sending_gd.append(client_average_weight_gd)
        model = get_model(dataset)
        model.set_weights(client_average_weight_gd)
        model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
        result=model.evaluate(X_valid, y_valid)
        accuracy=result[1]
        print('#######-----Gradient diversion acccuracy for round ', index1, 'is ', accuracy, ' ------########')
        accuracy_list_gd.append(accuracy)

        # bf posioning setting averaging
        client_average_weight_bf=model_average(client_weights_tobe_averaged_bf)
        client_weight_for_sending_bf.append(client_average_weight_bf)
        model = get_model(dataset)
        model.set_weights(client_average_weight_bf)
        model.compile(loss='categorical_crossentropy',optimizer="adam",metrics=['accuracy'])
        result=model.evaluate(X_valid, y_valid)
        accuracy=result[1]
        print('#######-----Bruteforce acccuracy for round ', index1, 'is ', accuracy, ' ------########')
        accuracy_list_bf.append(accuracy)

        
    return accuracy_list, accuracy_list_t100, accuracy_list_gd, accuracy_list_bf


if __name__=="__main__":

    accuracy_list_benign, accuracy_list_t100, accuracy_list_gd, accuracy_list_bf = train_server(COMMUNICATION_ROUNDS, EPOCHS_PER_CLIENT, 32, 0.01)
    
    with open('mnist_benign_acc_3tests_50.npy', 'wb') as f:
        np.save(f, accuracy_list_benign)
    f.close()

    with open('mnist_targetted_acc_3tests_50.npy', 'wb') as f:
        np.save(f, accuracy_list_t100)
    f.close()

    with open('mnist_gd_acc_3tests_50.npy', 'wb') as f:
        np.save(f, accuracy_list_gd)
    f.close()

    with open('mnist_bf_acc_3tests_50.npy', 'wb') as f:
        np.save(f, accuracy_list_bf)
    f.close()

