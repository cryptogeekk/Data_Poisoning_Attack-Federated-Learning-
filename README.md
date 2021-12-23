# Inverted Gradient Poisoning Attack on  Federated Learning.

Gradient Poisoning attack is a novel data poisoning attack that works on the principle of gradient divergence. Earlier, poisoning attacks like random poisoning and targeted works by crafting a predefined mapping between a true label and a poisoned label. These attacks target the direction of gradients in every SGD iteration. The target creates randomness in the direction of gradient during convergence and disturbs the minima of a given loss function.

In our attack, we exploit this randomness of the gradient to a greater extent. Instead of just creating simple randomness, we almost reverse the direction of the gradients during every SGD iteration. Reversing the gradients can be done with the inverted loss function, which is [described here.](https://drive.google.com/file/d/1e6NCKgv8UB9BUWc6O1_9623XY-Nnfbrt/view) 

Please refer to the following diagram, which gives a better understanding between the inverted gradient attack and other attacks.

![alt text](https://github.com/cryptogeekk/Data_Poisoning_Attack-Federated-Learning-/blob/main/inverted.png)



We have tested our attack on three benchmark datasets, i.e., MNIST, Fashion MNIST, CIFAR 10. We have found that our attack is **1.63 times stronger** than the targeted attack, which is itself **3.2 times** stronger than a random label attack and **11.6 times** than a random whole attack under the same setting.

## Usage

Clone the whole repository

Install the requirements.
python server.py
```bash
pip install requirements.txt
```
Running the server.py file will automatically instantiate all the server.

```bash
python server.py
```
# Result
We run our experiment under ten different nodes in three different malicious settings. If *f* denotes the fraction of malicious adversaries then, *f=0.1, f=0.3, and f=0.5*.
![alt text](https://github.com/cryptogeekk/Data_Poisoning_Attack-Federated-Learning-/blob/main/Data-Poisoning-Result.jpeg)
