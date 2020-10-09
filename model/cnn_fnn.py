"""
convoluational neural network (CNN): convolutional layers + fully connected layers
"""
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
from .relunet import ReluNet


class CnnFnn(nn.Module):

    """ Class for CNN model
    """
    def __init__(self, num_var,
                 input_dim, output_dim,
                 kernel_size=9, stride=5,
                 hidden_dim=100, num_layers=2,
                 num_epochs=100, learning_rate=1e-3):
        """ Initilize CNN model
        Args:
                num_var: int -- number of covariates as input, one CNN for each covariate
                input_dim: int -- dimension of the input for fully connected layers after apply cnn
                output_dim: int -- dimension of the output feature
                kernel_size: int -- Size of the convolving kernel
                srtide: int -- Stride of the convolution
                hidden_dim: int -- number of hidden units for fully connected layers
                num_layers: int -- number of hidden layers for fully connected layers
                num_epochs: int -- number of epochs to train
                learning_rate: float -- learning rate for ADAM
        """

        super(CnnFnn, self).__init__()
        #len(args.covariates_us)+len(args.covariates_global)+len(args.covariates_sea)
        self.num_var = num_var
        self.kernel_size = kernel_size
        self.stride = stride

        self.cnns = nn.ModuleList([nn.Sequential(nn.Conv3d(1, 1, (1, self.kernel_size, self.kernel_size),
                                                           (1, self.stride, self.stride)),
                                                 nn.ReLU(inplace=True)) for i in range(self.num_var)])
                                                 # nn.MaxPool3d(kernel_size=3,stride=3))for i in range(self.num_var)])

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        self.fnn = ReluNet(input_dim=self.input_dim * 4,
                           output_dim=self.output_dim,
                           hidden_dim=self.hidden_dim,
                           num_layers=self.num_layers,
                           num_epochs=self.num_epochs)

    def forward(self, src, device):
        """ Forward function
        """

        for i in range(self.num_var):

            input_x = torch.as_tensor(src[i]).float()
            input_x = input_x.to(device)
            x_out = self.cnns[i](input_x).squeeze(axis=1)  # squeeze the dimension 1: cnn input dim?
            x_flat = x_out.view(x_out.shape[0], x_out.shape[1], -1)
            if i == 0:
                x_append = x_flat
            else:
                x_append = torch.cat((x_append, x_flat), axis=-1)

        x_append = x_append.view(x_append.shape[0], -1)

        self.fnn = self.fnn.to(device)
        output = self.fnn(x_append)

        return output

    def fit(self, train_loader, device):
        """ Fit function to CNN
        """

        optimizer = optim.Adam(self.parameters(), self.learning_rate)

        criterion = torch.nn.MSELoss(reduction='mean')  # sum of the error for all element in the batch

        max_epoch = self.num_epochs

        for epoch in range(max_epoch):

            self.train()

            train_epoch_loss = 0

            for j, (src, trg) in enumerate(train_loader):

                trg = torch.as_tensor(trg).float().to(device)

                train_output = self.forward(src, device)

                loss = criterion(train_output, trg)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_epoch_loss += loss.item()

            print('Epoch: {}/{} Train Loss: {:.4f}'.format(epoch, max_epoch, train_epoch_loss/(j+1)))

    def fit_cv(self, train_loader, val_src, val_trg, device):
        """ Fit function for hyper-parameter tuning
        """
        val_trg = torch.as_tensor(val_trg).float().to(device)

        optimizer = optim.Adam(self.parameters(), self.learning_rate)

        # the mean error of all element in the batch
        criterion = torch.nn.MSELoss(reduction='mean')
        # array to story training-validation history
        history = np.zeros((self.num_epochs, 2))

        max_epoch = self.num_epochs

        for epoch in range(max_epoch):
            self.train()
            train_epoch_loss = 0

            for j, (src, trg) in enumerate(train_loader):

                trg = torch.as_tensor(trg).float().to(device)

                train_output = self.forward(src, device)

                loss = criterion(train_output, trg)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_epoch_loss += loss.item()

            # on validation set
            self.eval()
            val_output = self.forward(val_src, device)
            loss = criterion(val_output, val_trg)
            val_epoch_loss = loss.item()
            history[epoch] = [train_epoch_loss/(j+1), val_epoch_loss]

            print('Epoch: {}/{} Train Loss: {:.4f} Validation Loss:{:.4f}'
                  .format(epoch, self.num_epochs, train_epoch_loss/(j+1), val_epoch_loss))

        return history

    # make prediction
    def predict(self, src, device):
        """ Predict function for trained CNN-FNN model to predict
        """
        self.eval()

        with torch.no_grad():
            output = self.forward(src, device)

        return output.detach().cpu().numpy()


def get_input_dim(X, num_var, stride, kernel_size):
    input_dim = 0
    for i in range(num_var):

        W = X[i].shape[-1]
        H = X[i].shape[-2]

        S = stride
        K = kernel_size

        input_dim += (int((W-K)/S)+1)*(int((H-K)/S)+1)

    return input_dim
