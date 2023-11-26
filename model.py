import numpy as np

class Perceptron:
    def __init__(self, model_parameters, input_dim, output_dim):
        self.model_parameters = model_parameters
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = self.create_layers()

    @staticmethod
    def initiate_weights(input_dim, output_dim, is_init=False):
        weight_matrix = np.random.normal(size=(input_dim, output_dim))
        if is_init:
            weight_matrix[-1] = np.ones(shape=(1, output_dim))
        return weight_matrix

    def create_layers(self):
        layer_info = dict()
        input_dim = self.input_dim
        is_init = True
        for idx, (layer_name, layer_dim) in enumerate(self.model_parameters.items()):
            layer_info[layer_name] = {
                'weights': self.initiate_weights(input_dim, layer_dim, is_init=is_init),
                'layer_out': np.zeros(shape=(layer_dim, 1)),
                'layer_in': np.zeros(shape=(input_dim, 1)),
            }
            input_dim = layer_dim
            is_init = False
            if idx == len(self.model_parameters) - 1:
                layer_info[f'l{idx+1}'] = {
                    'weights': self.initiate_weights(input_dim, self.output_dim),
                    'layer_out': np.zeros(shape=(self.output_dim, 1)),
                    'layer_in': np.zeros(shape=(input_dim, 1)),
                }
        return layer_info

    def forward_step(self, mat_left, mat_right):
        layer_out = np.dot(mat_left, mat_right)

        return layer_out

    def forward(self, input_data):
        last_layer = list(self.layers.keys())[-1]
        layer_input = np.reshape(input_data, newshape=(input_data.shape[0], 1))
        outputs = dict()
        inputs = dict()
        for layer, information in self.layers.items():
            weight_matrix = information['weights']
            outputs[layer] = self.forward_step(weight_matrix.T, layer_input)
            inputs[layer] = layer_input
            layer_input = outputs[layer]

        for layer in self.layers.keys():
            self.layers[layer]['layer_out'] = outputs[layer]
            self.layers[layer]['layer_in'] = inputs[layer]
        return self.layers[last_layer]['layer_out']

    def create_weight_vectors(self, weight_matrix):
        weight_vector = np.reshape(weight_matrix, newshape=(1, weight_matrix.shape[0] * weight_matrix.shape[1]))
        return weight_vector

    def generate_weight_matrix(self, weight_vector, matrix_shape):
        weight_matrix = np.reshape(weight_vector, newshape=matrix_shape)
        return weight_matrix
    def create_impact_matrix(self, layer_id):
        layer_idx = f'l{layer_id}'
        in_shape = self.layers[layer_idx]['layer_in'].shape
        out_shape = self.layers[layer_idx]['layer_out'].shape
        impact_matrix = np.dot(np.eye(out_shape[0]), self.layers[layer_idx]['layer_in'][0, 0])
        for idx in range(1, in_shape[0]):
            new_rep = np.dot(np.eye(out_shape[0]), self.layers[layer_idx]['layer_in'][idx, 0])
            impact_matrix = np.concatenate((impact_matrix, new_rep), axis=1)

        return impact_matrix

    def update_layer(self, layer_id):
        num_layers = len(self.layers.keys())
        update = self.create_impact_matrix(layer_id)
        if f'l{layer_id + 1}' in self.layers.keys():
            for l_idx in range(layer_id + 1, num_layers):
                update = np.dot(self.layers[f'l{l_idx}']['weights'].T, update)
        return update

    def update_initial_layer(self, loss_grad):
        update = self.update_layer(0)
        weight_vector = self.create_weight_vectors(self.layers['l0']['weights'])
        weight_vector = weight_vector - 0.001 * np.dot(loss_grad, update)
        # print(weight_vector)
        weight_matrix = self.generate_weight_matrix(weight_vector, self.layers['l0']['weights'].shape)
        return weight_matrix

    def update(self, loss_grad, lr=0.001):

        num_layers = len(self.layers)
        updates = dict()
        for idx in range(num_layers-1, 0, -1):
            weight_vector = self.create_weight_vectors(self.layers[f'l{idx}']['weights'])
            update = lr * self.update_layer(layer_id=idx)
            weight_vector = weight_vector - np.dot(loss_grad, update)
            updates[f'l{idx}'] = self.generate_weight_matrix(
                weight_vector, self.layers[f'l{idx}']['weights'].shape
            )
        updates['l0'] = self.update_initial_layer(loss_grad=loss_grad)

        for idx in range(num_layers):

            self.layers[f'l{idx}']['weights'] = updates[f'l{idx}']




