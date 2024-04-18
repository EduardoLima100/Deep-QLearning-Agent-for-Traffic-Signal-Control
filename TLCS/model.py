import os
import torch
from torch import nn
import torch.optim as optim
from torchviz import make_dot
from torch.autograd import Variable
import sys

class TrainModel:
    def __init__(self, num_layers, width, batch_size, learning_rate, input_dim, output_dim, model_path=None):
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._batch_size = batch_size
        self._learning_rate = learning_rate

        if model_path is None:
            # build a new model
            self._model = self._build_model(num_layers, width)
        else:
            # load an existing model
            self._model = self._load_model(model_path)

        self._model.train()
        self._criterion = nn.MSELoss()
        self._optimizer = optim.Adam(self._model.parameters(), lr=self._learning_rate)

    def _build_model(self, num_layers, width):
        """
        Build and compile a fully connected deep neural network
        """
        dropout_rate = 0.0  # Defina a taxa de dropout aqui.

        model = nn.Sequential(
            nn.Linear(self._input_dim, width),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Adicione a camada de Dropout aqui
            *[
                nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Dropout(dropout_rate)) for _ in range(num_layers)
            ],
            nn.Linear(width, self._output_dim)
        )

        return model


    def _load_model(self, model_folder_path):
        """
        Load the model stored in the folder specified by the model number, if it exists
        """
        model_file_path = os.path.join(model_folder_path, 'trained_model.pt')

        if os.path.isfile(model_file_path):
            loaded_model = torch.load(model_file_path)
            return loaded_model
        else:
            sys.exit("Model file not found")

    def predict_one(self, state):
        """
        Predict the action values from a single state
        """
        state = torch.FloatTensor(state).unsqueeze(0)
        return self._model(state).detach().numpy()

    def predict_batch(self, states):
        """
        Predict the action values from a batch of states
        """
        states = torch.FloatTensor(states)
        return self._model(states).detach().numpy()

    def train_batch(self, states, q_sa):
        """
        Train the nn using the updated q-values
        """
        states = torch.FloatTensor(states)
        q_sa = torch.FloatTensor(q_sa)

        predictions = self._model(states)
        loss = self._criterion(predictions, q_sa)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()

    def save_model(self, path):
        """
        Save the current model in the folder as pt file and a model architecture summary as png
        """
        torch.save(self._model, os.path.join(path, 'trained_model.pt'))
        x = Variable(torch.randn(1,self._input_dim))
        y = self._model(x)
        make_dot(y.mean(), params=dict(self._model.named_parameters())).render(os.path.join(path, 'model_structure'), format="png")


class TestModel:
    def __init__(self, input_dim, model_path):
        self._input_dim = input_dim
        self._model = self._load_my_model(model_path)
        self._model.eval()

    def _load_my_model(self, model_folder_path):
        """
        Load the model stored in the folder specified by the model number, if it exists
        """
        model_file_path = os.path.join(model_folder_path, 'trained_model.pt')
        
        if os.path.isfile(model_file_path):
            loaded_model = torch.load(model_file_path)
            return loaded_model
        else:
            sys.exit("Model number not found")

    def predict_one(self, state):
        """
        Predict the action values from a single state
        """
        state = torch.FloatTensor(state).unsqueeze(0)
        return self._model(state).detach().numpy()
