import os
import pickle
import numpy as np
import matplotlib.pyplot as plt



class Visualization:

    def __init__(self, result_folder, experiment_number, dataset, model):
        self.folder = result_folder
        self.model = model
        self.experiment_folder = os.path.join(self.folder, f'experiment_{experiment_number}')
        self.dataset = dataset

    def plot_dataset(self):

        data = [each['data'][:-1] for each in self.dataset]
        targets = [each['target'] for each in self.dataset]

        plt.scatter(data, targets)
        plt.show()

    def fit_model(self, given_epoch):
        model_dir = os.path.join(self.experiment_folder, 'model')
        model_data = os.path.join(model_dir, f'model_{given_epoch}.pickle')
        with open(model_data, 'rb') as model_dict:
            data = pickle.load(model_dict)

        for layer_name, info in data.items():
            self.model.layers[layer_name] = info

        data = list()
        preds = list()
        targets = list()
        for datapoint in self.dataset:
            data.append(datapoint['data'][:-1])
            targets.append(datapoint['target'])
            out = self.model.forward(datapoint['data'])
            preds.append(out[0])

        plt.scatter(data, targets)
        plt.plot(data, preds, color='r')
        plt.show()

    def plot_result(self):
        results_file = os.path.join(self.experiment_folder, 'results.pickle')

        with open(results_file, 'rb') as data:
            results = pickle.load(data)

        epochs = list(results.keys())
        loss = list(results.values())
        plt.plot(epochs, loss)
        plt.show()