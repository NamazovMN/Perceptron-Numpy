from dataset import RegressionData
from model import Perceptron
from train import Train
from visualize import Visualization

def __main__():
    num_data = 2000
    input_range = (1, 10)
    target_range = (2, 8)
    sample_size = 200
    reg_data = RegressionData(num_data=num_data, input_range=input_range,
                              target_range=target_range, sample_size=sample_size)
    model_parameters = {
        'l0': 2,
        'l1': 3,
        'l2': 2
    }
    perceptron = Perceptron(model_parameters, 2, 1)
    save_directory = 'model_results'
    # trainer = Train(perceptron, reg_data, save_directory)
    # trainer.train(100, 0.0003, 3)

    visualization = Visualization(save_directory, 3, reg_data, perceptron)
    # visualization.plot_dataset()
    # for epoch in range(100):
    #     if epoch % 10 == 0:
    #         visualization.fit_model(epoch)

    visualization.plot_result()

if __name__ == '__main__':
    __main__()

