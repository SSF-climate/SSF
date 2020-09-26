
"""
Fully connected neural network with relu activation

"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class ReluNet(nn.Module):
    """ Fully connected neural network with relu activation
    """
    def __init__(self, input_dim, output_dim, hidden_dim=100, num_layers=2,
                 num_epochs=100, learning_rate=0.001, threshold=0.1):

        """ Initilize a Neural Network with Relu activation
        Args:
                input_dim: int -- dimension of the input feature
                output_dim: int -- dimension of the output feature
                hidden_dim: int -- number of hidden units at each layer
                num_layers: int -- number of hidden layers
                num_epochs: int -- number of epochs to train
                learning_rate: float -- learning rate for ADAM
                threshold: float -- the threshold to stop
        """
        super(ReluNet, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_epochs = num_epochs
        self.threshold = threshold
        self.learning_rate = learning_rate


        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        self.layers.append(nn.ReLU())
        for i in range(num_layers-1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())
        # output layer
        self.layers.append(nn.Linear(hidden_dim, output_dim))


    def forward(self, x):
        """ Forward function
        """
        out = self.layers[0](x)
        for i in range(1, len(self.layers)):
            out = self.layers[i](out)

        return out


    def fit(self, train_loader, device):
        """ Fit function to train the Relu network
        """

        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        criterion = torch.nn.MSELoss(reduction='mean') 
        max_epoch = self.num_epochs

        for epoch in range(max_epoch):

            self.train()
            train_epoch_loss = 0

            for i, (src, trg) in enumerate(train_loader):
                src = torch.as_tensor(src).float()
                src = src.to(device)
                trg = torch.as_tensor(trg).float()
                trg = trg.to(device)

                train_output = self.forward(src) # 1x197

                loss = criterion(train_output, trg)

                loss.backward()
                optimizer.step()
                train_epoch_loss += loss.item()


            print('Epoch: {}/{} Train Loss: {:.4f}'
                  .format(epoch, max_epoch, train_epoch_loss/(i+1)))

            if train_epoch_loss/(i+1) < self.threshold:
                break
            

    def fit_cv(self, train_loader, val_src, val_trg, device):
        """ Fit function for hyper-parameter tuning
        """

        val_src = torch.as_tensor(val_src).float()
        val_trg = torch.as_tensor(val_trg).float()

        val_src = val_src.to(device)
        val_trg = val_trg.to(device)

        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        criterion = torch.nn.MSELoss(reduction='mean')

        history = np.zeros((self.num_epochs, 2))

        for epoch in range(self.num_epochs):
            self.train()
            train_epoch_loss = 0
            for i, (src, trg) in enumerate(train_loader):
                src = torch.as_tensor(src).float()
                src = src.to(device)
                trg = torch.as_tensor(trg).float()
                trg = trg.to(device)

                train_output = self.forward(src) # 1x197

                loss = criterion(train_output, trg)

                loss.backward()
                optimizer.step()
                train_epoch_loss += loss.item()


            # on validation set
            self.eval()
            val_output = self.forward(val_src)
            loss = criterion(val_output, val_trg)
            val_epoch_loss = loss.item()
            history[epoch] = [train_epoch_loss/(i+1), val_epoch_loss]

            print('Epoch: {}/{} Train Loss: {:.4f} Validation Loss:{:.4f}'
                  .format(epoch, self.num_epochs, train_epoch_loss/(i+1), val_epoch_loss))

            if train_epoch_loss/(i+1) < self.threshold:
                break

        return history[:epoch]



    # make prediction
    def predict(self, src, device):
        """ Predict function for trained Relu network to predict
        """

        self.eval()
        src = torch.as_tensor(src).float()
        src = src.to(device)

        return self.forward(src).detach().cpu().numpy()
