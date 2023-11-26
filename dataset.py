import math

import numpy as np
import random


class RegressionData:
    def __init__(self, num_data, input_range, target_range, sample_size):
        self.num_data = num_data
        self.input_range = input_range
        self.target_range = target_range
        self.sample_size = sample_size
        self.input_data, self.target_data = self.process()

    def create_data(self, is_input=False):
        if is_input:
            difference = self.input_range[1] - self.input_range[0]
        else:
            difference = self.target_range[1] - self.target_range[0]

        data_range = self.input_range if is_input else self.target_range
        step_size = difference / self.num_data

        data = [data_range[0] + idx * step_size for idx in range(self.num_data + 1)]

        return data

    @staticmethod
    def add_bias(input_data, bias_value=0.5):
        result_data = np.zeros(shape=(input_data.shape[0] + 1, input_data.shape[-1]))
        result_data[:-1, :] = input_data
        result_data[-1, :] = np.array([[bias_value] * input_data.shape[-1]])
        return result_data

    def sample_and_sort(self, reverse=True, is_input=False):
        raw_data = self.create_data(is_input=is_input)
        sampled_data = random.sample(raw_data, self.sample_size)
        sampled_data.sort(reverse=reverse)

        return np.array([sampled_data])

    def process(self):
        x_data = self.sample_and_sort(reverse=False, is_input=True)
        target_data = self.sample_and_sort(reverse=False, is_input=False)
        input_data = self.add_bias(x_data)

        return input_data, target_data

    def __getitem__(self, idx):
        return {
            'data': self.input_data[:, idx],
            'target': self.target_data[:, idx]
        }

    def __len__(self):
        return len(self.input_data)