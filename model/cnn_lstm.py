"""
CNN-LSTM model
"""
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
from .recnet import LSTM




 
class CnnLSTM(nn.Module):
    """ Class for CNN-LSTM model
    """
    def __init__(self,
                 num_var, input_dim, output_dim,
                 kernel_size=9, stride=5,
                 hidden_dim=100, num_lstm_layers=2,
                 num_epochs=100, learning_rate=1e-3):
        """ Initilize CNN-LSTM model
        Args:
                num_var: int -- number of covariates as input, one CNN for each covariate
                input_dim: int -- dimension of the input for LSTM after apply cnn
                output_dim: int -- dimension of the output feature
                kernel_size: int -- Size of the convolving kernel
                srtide: int -- Stride of the convolution
                hidden_dim: int -- number of hidden units for LSTM
                num_lstm_layers: int -- number of hidden layers for LSTM
                num_epochs: int -- number of epochs to train
                learning_rate: float -- learning rate for ADAM
        """
        super(CnnLSTM, self).__init__()

        #len(args.covariates_us)+len(args.covariates_global)+len(args.covariates_sea)
        self.num_var = num_var
        self.kernel_size = kernel_size
        self.stride = stride

        self.cnns = nn.ModuleList([nn.Sequential(nn.Conv3d(1, 1, (1, self.kernel_size, self.kernel_size),
                                                           (1, self.stride, self.stride)),
                                                 nn.ReLU(inplace=True)) for i in range(self.num_var)])
                                                 #nn.MaxPool3d(kernel_size=3,stride=3))for i in range(self.num_var)])



        self.input_dim = input_dim#self.get_input_dim(src)
        self.output_dim = output_dim#trg.shape[-1]

        self.hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs


        self.lstm = LSTM(input_dim=self.input_dim,
                         output_dim=self.output_dim,
                         hidden_dim=self.hidden_dim,
                         num_layers=self.num_lstm_layers,
                         learning_rate=self.learning_rate,
                         num_epochs=self.num_epochs)



        
    def forward(self, src, device):
        """ Forward function
        """

        for i in range(self.num_var):

            input_x = torch.as_tensor(src[i]).float()
            input_x = input_x.to(device)

            # squeeze the dimension 1: cnn input dim
            x_out = self.cnns[i](input_x).squeeze(axis=1)
            x_flat = x_out.view(x_out.shape[0], x_out.shape[1], -1)
            if i == 0:
                x_append = x_flat
            else:
                x_append = torch.cat((x_append, x_flat), axis=-1)


        output = self.lstm(x_append, device)

   
        return output


    def fit(self, train_loader, device):
        """ Fit function to train CNN-LSTM
        """

        optimizer = optim.Adam(self.parameters(), self.learning_rate)
        # take the mean error for all element in the batch,
        # can be changed to 'sum'
        criterion = torch.nn.MSELoss(reduction='mean') 

        max_epoch = self.num_epochs


        for epoch in range(max_epoch):      
            

            self.train()

            train_epoch_loss = 0           

            for j, (src, trg) in enumerate(train_loader):

                trg = torch.as_tensor(trg).float().to(device)
                #src = torch.as_tensor(src).float().to(device)
                
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
        # take the mean error for all element in the batch
        criterion = torch.nn.MSELoss(reduction='mean')
        # array to story training-validation history
        history = np.zeros((self.num_epochs, 2)) 

        max_epoch = self.num_epochs


        for epoch in range(max_epoch):
            #on training set
            self.train()

            train_epoch_loss = 0           

            for i, (src, trg) in enumerate(train_loader):

                trg = torch.as_tensor(trg).float().to(device)
                #src = torch.as_tensor(src).float().to(device)
                
                train_output = self.forward(src, device)

                loss = criterion(train_output, trg)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


                train_epoch_loss += loss.item()

            #on validation set
            self.eval()
            val_output = self.forward(val_src, device)
            loss = criterion(val_output, val_trg)
            val_epoch_loss = loss.item()
            history[epoch] = [train_epoch_loss/(i+1), val_epoch_loss]

            print('Epoch: {}/{} Train Loss: {:.4f} Validation Loss:{:.4f}'
                  .format(epoch, self.num_epochs, train_epoch_loss/(i+1), val_epoch_loss))

        return history



    # make prediction
    def predict(self, src, device):
        """ Predict function for trained CNN-LSTM model to predict
        """

        self.eval()

        with torch.no_grad():
            output = self.forward(src, device)

        return output.detach().cpu().numpy() 
