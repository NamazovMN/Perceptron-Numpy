import os
from tqdm import tqdm
import numpy as np
import pickle
class Train:
    def __init__(self, model, dataset, save_dir):
        self.model = model
        self.dataset = dataset
        self.check_dir(save_dir)
        self.save_dir = save_dir

    def check_dir(self, directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    def loss_fn(self, target, prediction):
        difference = target - prediction
        return np.linalg.norm(difference) ** 2 / target.shape[0]

    def loss_grad_fn(self, target, prediction):
        return -2 * (target - prediction)

    def train(self, epochs, learning_rate, exp_num):
        experiment_name = f'experiment_{exp_num}'
        experiment_folder = os.path.join(self.save_dir, experiment_name)

        model_folder = os.path.join(experiment_folder, 'model')
        self.check_dir(model_folder)
        results_file = os.path.join(experiment_folder, 'results.pickle')
        results_dict = dict()
        for epoch in range(epochs):
            iterator = tqdm(iterable=enumerate(self.dataset), total=len(self.dataset))
            epoch_loss = 0
            for d_idx, datapoint in iterator:
                out = self.model.forward(datapoint['data'])
                loss_val = self.loss_fn(datapoint['target'], out)
                epoch_loss += loss_val
                loss_grad = self.loss_grad_fn(datapoint['target'], out)

                self.model.update(loss_grad=loss_grad, lr=learning_rate)
                iterator.set_description(desc=f'Training: {epoch + 1}, Loss: {epoch_loss}')
            average_loss = epoch_loss / len(self.dataset)
            print(f'\nEpoch Result: Train Loss: {average_loss: .4f}')
            results_dict[epoch] = average_loss
            model_parameters = self.model.layers
            parameters_file = os.path.join(model_folder, f'model_{epoch}.pickle')
            self.save_parameters(model_parameters, parameters_file, f"Model for epoch {epoch}")
        self.save_parameters(results_dict, results_file, 'Train results')

    def save_parameters(self, data, file_name, what):

        with open(file_name, 'wb') as save_data:
            pickle.dump(data, save_data)
        print(f'{what} was saved for this experiment')
        print('<<<<<<<<<<<<<<<<<<<<<<<<>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')

