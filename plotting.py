import numpy as np
import matplotlib.pyplot as plt


def plot(training_round):

    benign_accuracy=np.load('./mnist_benign_acc_3tests_50.npy')
    targeted_accuracy=np.load('./mnist_targetted_acc_3tests_50.npy')
    gd_accuracy=np.load('./mnist_gd_acc_3tests_50.npy')
    bf_accuracy=np.load('./mnist_bf_acc_3tests_50.npy')

    
    
    plt.plot(training_round,targeted_accuracy,'r',label='targetted')
    plt.plot(training_round,gd_accuracy,'g',label='gradient')
    plt.plot(training_round,bf_accuracy,'c',label='brute force')
    plt.plot(training_round,benign_accuracy,'b',label='benign')
#     plt.plot(training_round,targeted_accuracy05,'m',label='50% TA')
    plt.xlabel('training_round')
    plt.ylabel('accuracy')
    plt.legend(loc="best")
    plt.savefig('acccc.png')
    
    plt.show()

arr = np.arange(0,100)
plot(arr)