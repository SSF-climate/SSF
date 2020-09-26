"""LSTM as a basic building block to predict temp/precip of 2-4 weeks in advance
"""
import torch.optim as optim
import torch.nn as nn
import torch


class LSTM(nn.Module):
    """Class for LSTM 
    """

    def __init__(self,
                 input_dim, output_dim=1,
                 hidden_dim=10, num_layers=2,
                 learning_rate=1e-3, num_epochs=100):
        """ Initilize LSTM model
        Args:
                input_dim: int -- dimension of expected features in input X
                output_dim: int -- dimension of the output feature,
                                   for regrssion problem, the output_dim = 1
                hidden_dim: int -- number of hidden units for LSTM
                num_layers: int -- number of stacked recurrent layers.
                num_epochs: int -- number of epochs to train
                learning_rate: float -- learning rate for ADAM
        """
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers, batch_first=True)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)


    def forward(self, src, device):
        """ Forward pass through LSTM layer

        Args:
                src: of shape [batch, seq_len, input_size]
        """

        src = torch.as_tensor(src).float().to(device)


        # Set initial hidden and cell states 
        h0 = torch.zeros(self.num_layers, src.shape[0], self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers, src.shape[0], self.hidden_dim).to(device)

        # shape of lstm_out: [batch_size, input_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both have shape (num_layers, batch_size, hidden_dim).
        lstm_out, self.hidden = self.lstm(src, (h0, c0)) 
        
        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        #print(lstm_out.size())
        y_pred = self.linear(lstm_out[:, -1, :].view(src.shape[0], -1))
        return y_pred

    def fit(self, src, trg, device):
        """ Fit function to train LSTM
        """

        src = torch.as_tensor(src).float().to(device)
        trg = torch.as_tensor(trg).float().to(device)


        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        criterion = torch.nn.MSELoss(reduction='mean')

        for epoch in range(self.num_epochs):
            self.train()
            # Initialise hidden state
            # Don't do this if you want your LSTM to be stateful
            # mdl.hidden = mdl.init_hidden()

            # Forward pass
            y_pred = self.forward(src, device)

            loss = criterion(y_pred, trg)

            # Zero out gradient, else they will accumulate between epochs
            optimizer.zero_grad()

            # Backward pass
            loss.backward(retain_graph=True)

            # Update parameters
            optimizer.step()

            print('Epoch: {}/{} Train Loss: {:.4f}'.format(epoch, self.num_epochs, loss.item()))


    def predict(self, src):
        """ Predict function for trained LSTM model to make prediction
        """

        src = torch.as_tensor(src).float()

        self.eval()

        return self.forward(src)
