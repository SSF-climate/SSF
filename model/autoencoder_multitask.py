import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    The class for LSTM encoder
    """

    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.):
        """ Initilize LSTM encoder
        Args:
        input_dim: int -- the size/dimensionality of the vectors that will be input to the encoder
        hidden_dim: int -- the dimensionality of the hidden and cell states
        num_layers: int -- the number of layers in the LSTM
        dropout: float -- the amount of dropout to use
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)

    def forward(self, src):
        """
        Forward function, return hidden and cell
        """

        # src = [batch size, time, dim]

        outputs, (hidden, cell) = self.lstm(src)

        # outputs = [src sent len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]
        # outputs are always from the top hidden layer

        return hidden, cell


class Encoder_all_seq(nn.Module):
    """
        The class for LSTM encoder with the final output as the combination of the output for each date
    """

    def __init__(self, input_dim, hidden_dim, num_layers=2, dropout=0.):
        """ Initilize LSTM encoder
        Args:
        input_dim: int -- the size/dimensionality of the vectors that will be input to the encoder
        hidden_dim: int -- the dimensionality of the hidden and cell states
        num_layers: int -- the number of layers in the LSTM
        dropout: float -- the amount of dropout to use
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)

    def forward(self, src):
        """
        Forward function, return output from all steps in the input sequence
        """
        #  src = [batch size, time, dim]
        outputs, (hidden, cell) = self.lstm(src)
        return outputs


class Decoder(nn.Module):
    """
    The class for LSTM decoder
    """
    def __init__(self, output_dim, hidden_dim, num_layers, dropout):
        """ Initilize LSTM decoder
        Args:
        output_dim: int -- the size/dimensionality of the vectors that will be input to the decoder (output of encoder)
        hidden_dim: int -- the dimensionality of the hidden and cell states
        num_layers: int -- the number of layers in the LSTM
        dropout: float -- the amount of dropout to use
        """
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(output_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, input, hidden, cell):
        """Forward function
        """
        # input = [batch size]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]

        input = input.unsqueeze(1)  # batch first

        # input = [1, batch size]
        # embedded = [1, batch size, emb dim]

        output, (hidden, cell) = self.lstm(input, (hidden, cell))

        # output = [sent len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # sent len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]

        prediction = self.out(output.squeeze(0))

        # prediction = [batch size, output dim]

        return prediction, hidden, cell


class Rowwise_mult_layer(nn.Module):
    '''
    Rowwise multiply for AR, for each target location, there is one vector of coefficients for the corresponding sequence from history
    '''
    def __init__(self, ar_dim, num_locations):
        """ Initilize the Rowwise multiplication layer
        Args:
        ar_dim: int -- the size/dimensionality of the vectors that will be input layer (the length of "local" historical sequence)
        num_locations: int -- the number of target locations
        """
        super().__init__()
        self.ar_dim = ar_dim
        self.num_locations = num_locations
        self.weights = nn.Parameter(torch.zeros(self.num_locations, self.ar_dim))

    def forward(self, src):
        """Forward function
        """
        return torch.sum(src * self.weights, axis=-1)

# Six models are listed below
# (1) Encoder (LSTM) Decoder (LSTM) with quad loss
# (2) Encoder (LSTM) Decoder (LSTM)
# (3) Encoder (LSTM) Decoder (FNN) - Last layer or all layers
# (4) Encoder (LSTM) Decoder (FNN) - all seq
# (5) Encoder (LSTM) Decoder (FNN) - all seq + AR
# (6) Encoder (LSTM) Decoder (FNN) - all seq + AR (Climate indexes)


class EncoderDecoderQuadLoss(nn.Module):
    """The class for an encoder (LSTM)-decoder(LSTM) model with quadratic loss
    """
    def __init__(self, input_dim, output_dim, hidden_dim=10, num_layers=2, learning_rate=0.01, decoder_len=18, threshold=0.1, num_epochs=100):
        """ Initilize an encoder (LSTM)-decoder(LSTM) model with quadratic loss
        Args:
        input_dim: int -- the size/dimensionality of the vectors that will be input for encoder
        output_dim: int -- the size/dimensionality of the vectors that will be input for decoder
        hidden_dim: int -- the dimensionality of the hidden and cell states
        num_layers: int -- the number of layers in the LSTM
        learning_rate: float -- learning learning_rate
        decoder_len: int -- the sequence lenght of decoder(LSTM)
        threshold: float -- the early stopping point (training loss < threshold) for training process
        num_epochs: int -- the maximum number of training epochs
        """
        super().__init__()

        self.input_dim = input_dim  # batch_size seq_length n_features
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.decoder_len = decoder_len
        self.threshold = threshold
        self.encoder = Encoder(self.input_dim, self.hidden_dim, self.num_layers, 0.0)
        self.decoder = Decoder(self.output_dim, self.hidden_dim, self.num_layers, 0.0)

        assert self.encoder.hidden_dim == self.decoder.hidden_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert self.encoder.num_layers == self.decoder.num_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, device):
        """
        Forward function
        """
        # src = [batch size,sent len, src] e.g.[5,10,1]
        # trg = [batch size,sent len, trg] e.g.[5,10,1]

        src = torch.as_tensor(src).float()
        src = src.to(device)

        batch_size = src.shape[0]  # batch_size
        seq_len = src.shape[1]

        # tensor to store decoder outputs
        outputs = torch.zeros(batch_size, self.decoder_len, self.output_dim).to(device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        # input = trg[:,0] # 5x10
        input = torch.zeros(batch_size, self.output_dim).to(device)

        for t in range(0, self.decoder_len):

            # insert input, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)

            # print(output.shape)
            output = output.squeeze(1)
            # output =  output.reshape(-1,1)#output.squeeze(1) # reduce dimension to num_batch x num_feature
            # place predictions in a tensor holding predictions for each token
            outputs[:, t, :] = output

            # input = output
            # print(input.size())
        return outputs[:, -1, :]

    def fit(self, train_loader, device):
        """ Fit function for model training
        """
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        for epoch in range(self.num_epochs):

            self.train()
            train_epoch_loss = 0

            for i, (src, trg, C) in enumerate(train_loader):
                src = torch.as_tensor(src).float()
                src = src.to(device)
                if len(src.size()) < 3:
                    src = src.view(1, -1)
                trg = torch.as_tensor(trg).float()
                trg = trg.to(device)
                C = torch.as_tensor(C).float()
                C = C.to(device)

                train_output = self.forward(src, device)  # 1x197

                # loss = criterion(output[:,1:,:], trg[:,1:,:])
                loss = quad_loss(train_output, trg, C)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_epoch_loss += loss.item()

            print('Epoch: {}/{} Train Loss: {:.4f}'.format(epoch, self.num_epochs, train_epoch_loss / (i + 1)))

            if train_epoch_loss / (i + 1) < self.threshold:
                break

    def fit_cv(self, train_loader, val_src, val_trg, device):
        """ Fit function for hyper-parameter tuning
        """
        val_src = torch.as_tensor(val_src).float()
        val_trg = torch.as_tensor(val_trg).float()
        val_src = val_src.to(device)
        val_trg = val_trg.to(device)

        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        criterion = torch.nn.MSELoss(reduction='mean')  # sum of the error for all element in the batch
        history = np.zeros((self.num_epochs, 3))

        for epoch in range(self.num_epochs):
            self.train()
            train_epoch_loss = 0
            epoch_sq_loss = 0
            for i, (src, trg, C) in enumerate(train_loader):
                src = torch.as_tensor(src).float()
                src = src.to(device)
                if len(src.size()) < 3:
                    src = src.view(1, -1)
                trg = torch.as_tensor(trg).float()
                trg = trg.to(device)
                C = torch.as_tensor(C).float()
                C = C.to(device)
                train_output = self.forward(src, device)  # 1x197

                # loss = criterion(output[:,1:,:], trg[:,1:,:])
                loss = quad_loss(train_output, trg, C)
                loss_sq = criterion(train_output, trg)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_epoch_loss += loss.item()
                epoch_sq_loss += loss_sq.item()

            # on validation set
            self.eval()
            val_output = self.forward(val_src, device)
            loss = criterion(val_output, val_trg)
            val_epoch_loss = loss.item()
            history[epoch] = [train_epoch_loss / (i + 1), epoch_sq_loss / (i + 1), val_epoch_loss]
            print('Epoch: {}/{} Train Loss: {:.4f} Square Train Loss: {:.4f} Validation Loss:{:.4f}'.format(epoch, self.num_epochs, train_epoch_loss / (i + 1), epoch_sq_loss / (i + 1), val_epoch_loss))
            if epoch_sq_loss / (i + 1) < self.threshold:
                break
        return history[:epoch]

    # make prediction
    def predict(self, src, device):
        """ Predict function for a trained model to predict
        """
        self.eval()
        src = torch.as_tensor(src).float()
        src = src.to(device)
        return self.forward(src, device).detach().cpu().numpy()


class EncoderDecoder(nn.Module):
    """The class for an encoder (LSTM)-decoder(LSTM) model with l2 loss
    """
    def __init__(self, input_dim, output_dim, hidden_dim=10, num_layers=2, learning_rate=0.01, decoder_len=18, threshold=0.1, num_epochs=100):
        """ Initilize an encoder (LSTM)-decoder(LSTM) model with l2 loss
        Args:
        input_dim: int -- the size/dimensionality of the vectors that will be input for encoder
        output_dim: int -- the size/dimensionality of the vectors that will be input for decoder
        hidden_dim: int -- the dimensionality of the hidden and cell states
        num_layers: int -- the number of layers in the LSTM
        learning_rate: float -- learning learning_rate
        decoder_len: int -- the sequence lenght of decoder(LSTM)
        threshold: float -- the early stopping point (training loss < threshold) for training process
        num_epochs: int -- the maximum number of training epochs
        """
        super().__init__()

        self.input_dim = input_dim  # batch_size seq_length n_features
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.decoder_len = decoder_len
        self.threshold = threshold

        self.encoder = Encoder(self.input_dim, self.hidden_dim, self.num_layers, 0.0)
        self.decoder = Decoder(self.output_dim, self.hidden_dim, self.num_layers, 0.0)

        assert self.encoder.hidden_dim == self.decoder.hidden_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert self.encoder.num_layers == self.decoder.num_layers, \
            "Encoder and decoder must have equal number of layers!"

    def forward(self, src, device):
        """Forward function
        """
        # src = [batch size,sent len, src] e.g.[5,10,1]
        # trg = [batch size,sent len, trg] e.g.[5,10,1]

        src = torch.as_tensor(src).float()
        src = src.to(device)
        batch_size = src.shape[0]  # batch_size
        seq_len = src.shape[1]

        # tensor to store decoder outputs
        outputs = torch.zeros(batch_size, self.decoder_len, self.output_dim).to(device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)

        # first input to the decoder is the <sos> tokens
        # input = trg[:,0] # 5x10
        input = torch.zeros(batch_size, self.output_dim).to(device)

        for t in range(0, self.decoder_len):

            # insert input, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)

            # print(output.shape)
            output = output.squeeze(1)
            # output =  output.reshape(-1,1)#output.squeeze(1) # reduce dimension to num_batch x num_feature
            # place predictions in a tensor holding predictions for each token
            outputs[:, t, :] = output

            # input = output
            # print(input.size())
        return outputs[:, -1, :]

    def fit(self, train_loader, device):
        """ Fit function for model training
        """
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        criterion = torch.nn.MSELoss(reduction='mean')  # sum of the error for all element in the batch

        for epoch in range(self.num_epochs):

            self.train()
            train_epoch_loss = 0

            for i, (src, trg) in enumerate(train_loader):
                src = torch.as_tensor(src).float()
                src = src.to(device)
                if len(src.size()) < 3:
                    src = src.view(1, -1)
                trg = torch.as_tensor(trg).float()
                trg = trg.to(device)

                train_output = self.forward(src, device)  # 1x197

                # loss = criterion(output[:,1:,:], trg[:,1:,:])
                loss = criterion(train_output, trg)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_epoch_loss += loss.item()

            print('Epoch: {}/{} Train Loss: {:.4f}'.format(epoch, self.num_epochs, train_epoch_loss / (i + 1)))

            if train_epoch_loss / (i + 1) < self.threshold:
                break

    def fit_cv(self, train_loader, val_src, val_trg, device):
        """ Fit function for hyper-parameter tuning
        """
        val_src = torch.as_tensor(val_src).float()
        val_trg = torch.as_tensor(val_trg).float()
        val_src = val_src.to(device)
        val_trg = val_trg.to(device)

        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        criterion = torch.nn.MSELoss(reduction='mean')  # sum of the error for all element in the batch
        history = np.zeros((self.num_epochs, 2))

        for epoch in range(self.num_epochs):
            self.train()
            train_epoch_loss = 0
            for i, (src, trg) in enumerate(train_loader):
                src = torch.as_tensor(src).float()
                train_output = self.forward(src, device)
                trg = torch.as_tensor(trg).float()
                trg = trg.to(device)
                loss = criterion(train_output, trg)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_epoch_loss += loss.item()
            # on validation set
            self.eval()
            val_output = self.forward(val_src, device)
            loss = criterion(val_output, val_trg)
            val_epoch_loss = loss.item()
            history[epoch] = [train_epoch_loss / (i + 1), val_epoch_loss]
            print('Epoch: {}/{} Train Loss: {:.4f} Validation Loss:{:.4f}'.format(epoch, self.num_epochs, train_epoch_loss / (i + 1), val_epoch_loss))
            if train_epoch_loss / (i + 1) < self.threshold:
                break
        return history[:epoch]

    # make prediction
    def predict(self, src, device):
        """ Predict function for a trained model to predict
        """
        self.eval()
        src = torch.as_tensor(src).float()
        src = src.to(device)

        return self.forward(src, device).detach().cpu().numpy()


class EncoderFNN(nn.Module):
    """The class for an encoder (LSTM)-decoder(FNN) model with l2 loss
    """
    def __init__(self, input_dim, output_dim, hidden_dim=10, num_layers=2, last_layer=True, seq_len=4, learning_rate=0.01, threshold=0.1, num_epochs=100):
        """ Initilize an encoder (LSTM)-decoder(FNN) model with l2 loss
        Args:
        input_dim: int -- the size/dimensionality of the vectors that will be input for encoder
        output_dim: int -- the size/dimensionality of the vectors that will be input for decoder
        hidden_dim: int -- the dimensionality of the hidden and cell states
        num_layers: int -- the number of layers in the LSTM
        last_layer: boolean -- if only take the output of the last layer in encoder as the input to decoder
        seq_len: int -- the length of input sequence
        learning_rate: float -- learning learning_rate
        threshold: float -- the early stopping point (training loss < threshold) for training process
        num_epochs: int -- the maximum number of training epochs
        """
        super().__init__()

        self.input_dim = input_dim  # batch_size seq_length n_features
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.threshold = threshold
        self.seq_len = seq_len

        self.encoder = Encoder(self.input_dim, self.hidden_dim, self.num_layers, 0.0)
        self.last_layer = last_layer
        # self.decoder = Decoder(self.output_dim,self.hidden_dim, self.num_layers, 0.0)
        if self.last_layer is True:
            self.out = nn.Linear(hidden_dim, output_dim)
        else:
            self.out = nn.Linear(hidden_dim * num_layers, output_dim)

        # assert self.encoder.hidden_dim == self.decoder.hidden_dim, \
        #     "Hidden dimensions of encoder and decoder must be equal!"
        # assert self.encoder.num_layers == self.decoder.num_layers, \
        #     "Encoder and decoder must have equal number of layers!"

    def forward(self, src, device):
        """Forward function
        """
        # src = [batch size,sent len, src] e.g.[5,10,1]
        # trg = [batch size,sent len, trg] e.g.[5,10,1]

        src = torch.as_tensor(src[:, -self.seq_len:, :]).float()
        # src = torch.as_tensor(src).float()
        src = src.to(device)
        batch_size = src.shape[0]  # batch_size
        seq_len = src.shape[1]

        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size,self.decoder_len,self.output_dim).to(device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)

        hidden_output = hidden.permute(1, 0, 2)

        if self.last_layer is True:
            hidden_output = hidden_output[:, -1, :]
            hidden_output = hidden_output.reshape(hidden_output.shape[0], self.hidden_dim)
        else:
            hidden_output = hidden_output.reshape(hidden_output.shape[0], hidden_output.shape[1] * hidden_output.shape[2])

        output = self.out(hidden_output)

        return output

    def fit(self, train_loader, device):
        """ Fit function for model training
        """
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        criterion = torch.nn.MSELoss(reduction='mean')  # sum of the error for all element in the batch

        for epoch in range(self.num_epochs):

            self.train()
            train_epoch_loss = 0

            for i, (src, trg) in enumerate(train_loader):
                src = torch.as_tensor(src).float()
                src = src.to(device)
                if len(src.size()) < 3:
                    src = src.view(1, -1)
                trg = torch.as_tensor(trg).float()
                trg = trg.to(device)

                train_output = self.forward(src, device)  # 1x197

                # loss = criterion(output[:,1:,:], trg[:,1:,:])
                loss = criterion(train_output, trg)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_epoch_loss += loss.item()

            print('Epoch: {}/{} Train Loss: {:.4f}'.format(epoch, self.num_epochs, train_epoch_loss / (i + 1)))

            if train_epoch_loss / (i + 1) < self.threshold:
                break

    def fit_cv(self, train_loader, val_src, val_trg, device):
        """ Fit function for hyper-parameter tuning
        """
        val_src = torch.as_tensor(val_src).float()
        val_trg = torch.as_tensor(val_trg).float()
        val_src = val_src.to(device)
        val_trg = val_trg.to(device)

        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        criterion = torch.nn.MSELoss(reduction='mean')  # sum of the error for all element in the batch
        history = np.zeros((self.num_epochs, 2))

        for epoch in range(self.num_epochs):
            self.train()
            train_epoch_loss = 0
            for i, (src, trg) in enumerate(train_loader):
                src = torch.as_tensor(src).float()
                train_output = self.forward(src, device)
                trg = torch.as_tensor(trg).float()
                trg = trg.to(device)
                loss = criterion(train_output, trg)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_epoch_loss += loss.item()
            # on validation set
            self.eval()
            val_output = self.forward(val_src, device)
            loss = criterion(val_output, val_trg)
            val_epoch_loss = loss.item()
            history[epoch] = [train_epoch_loss / (i + 1), val_epoch_loss]
            print('Epoch: {}/{} Train Loss: {:.4f} Validation Loss:{:.4f}'.format(epoch, self.num_epochs, train_epoch_loss / (i + 1), val_epoch_loss))
            if train_epoch_loss / (i + 1) < self.threshold:
                break
        return history[:epoch]

    # make prediction
    def predict(self, src, device):
        """ Predict function for a trained model to predict
        """
        self.eval()
        src = torch.as_tensor(src).float()
        src = src.to(device)

        return self.forward(src, device).detach().cpu().numpy()


class EncoderFNN_AllSeq(nn.Module):
    """The class for an encoder (LSTM)-decoder(FNN) model where the input of the decoder is the output of all steps in input sequence
    """
    def __init__(self, input_dim, output_dim, hidden_dim=10, num_layers=2, seq_len=4, linear_dim=100, learning_rate=0.01, dropout=0.1, threshold=0.1, num_epochs=100):
        """ Initilize an encoder (LSTM)-decoder(FNN) model where the input of the decoder is the output of all steps in input sequence
        Args:
        input_dim: int -- the size/dimensionality of the vectors that will be input for encoder
        output_dim: int -- the size/dimensionality of the vectors that will be input for decoder
        hidden_dim: int -- the dimensionality of the hidden and cell states
        num_layers: int -- the number of layers in the LSTM
        seq_len: int -- the length of input sequence
        linear_dim: int -- the dimensionality of the decoder FNN
        learning_rate: float -- learning learning_rate
        threshold: float -- the early stopping point (training loss < threshold) for training process
        num_epochs: int -- the maximum number of training epochs
        """
        super().__init__()

        self.input_dim = input_dim  # batch_size seq_length n_features
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.threshold = threshold
        self.seq_len = seq_len
        self.dropout = dropout
        self.linear_dim = linear_dim

        self.encoder = Encoder_all_seq(self.input_dim, self.hidden_dim, self.num_layers, self.dropout)

        self.out1 = nn.Linear(self.hidden_dim * self.seq_len, self.linear_dim)
        self.out2 = nn.Linear(self.linear_dim, self.output_dim)

        # assert self.encoder.hidden_dim == self.decoder.hidden_dim, \
        #     "Hidden dimensions of encoder and decoder must be equal!"
        # assert self.encoder.num_layers == self.decoder.num_layers, \
        #     "Encoder and decoder must have equal number of layers!"

    def forward(self, src, device):
        """Forward function
        """
        # src = [batch size,sent len, src] e.g.[5,10,1]
        # trg = [batch size,sent len, trg] e.g.[5,10,1]

        src = torch.as_tensor(src[:, -self.seq_len:, :]).float()
        src = src.to(device)
        batch_size = src.shape[0]  # batch_size
        seq_len = src.shape[1]

        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size,self.decoder_len,self.output_dim).to(device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        encoder_output = self.encoder(src)

        # encoder_output = encoder_output.permute(1,0,2)

        encoder_output = encoder_output.reshape(encoder_output.shape[0], encoder_output.shape[1] * encoder_output.shape[2])

        linear_output = self.out1(encoder_output)
        linear_output = F.relu(linear_output)
        output = self.out2(linear_output)

        return output

    def fit(self, train_loader, device):
        """ Fit function for model training
        """
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        criterion = torch.nn.MSELoss(reduction='mean')  # sum of the error for all element in the batch

        for epoch in range(self.num_epochs):

            self.train()
            train_epoch_loss = 0

            for i, (src, trg) in enumerate(train_loader):
                src = torch.as_tensor(src).float()
                src = src.to(device)
                if len(src.size()) < 3:
                    src = src.view(1, -1)
                trg = torch.as_tensor(trg).float()
                trg = trg.to(device)

                train_output = self.forward(src, device)  # 1x197

                # loss = criterion(output[:,1:,:], trg[:,1:,:])
                loss = criterion(train_output, trg)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_epoch_loss += loss.item()

            print('Epoch: {}/{} Train Loss: {:.4f}'.format(epoch, self.num_epochs, train_epoch_loss / (i + 1)))

            if train_epoch_loss / (i + 1) < self.threshold:
                break

    def fit_cv(self, train_loader, val_src, val_trg, device):
        """ Fit function for hyper-parameter tuning
        """
        val_src = torch.as_tensor(val_src).float()
        val_trg = torch.as_tensor(val_trg).float()
        val_src = val_src.to(device)
        val_trg = val_trg.to(device)

        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        criterion = torch.nn.MSELoss(reduction='mean')  # sum of the error for all element in the batch
        history = np.zeros((self.num_epochs, 2))

        for epoch in range(self.num_epochs):
            self.train()
            train_epoch_loss = 0
            for i, (src, trg) in enumerate(train_loader):
                src = torch.as_tensor(src).float()
                train_output = self.forward(src, device)
                trg = torch.as_tensor(trg).float()
                trg = trg.to(device)
                loss = criterion(train_output, trg)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_epoch_loss += loss.item()
            # on validation set
            self.eval()
            val_output = self.forward(val_src, device)
            loss = criterion(val_output, val_trg)
            val_epoch_loss = loss.item()
            history[epoch] = [train_epoch_loss / (i + 1), val_epoch_loss]
            print('Epoch: {}/{} Train Loss: {:.4f} Validation Loss:{:.4f}'.format(epoch, self.num_epochs, train_epoch_loss / (i + 1), val_epoch_loss))
            if train_epoch_loss / (i + 1) < self.threshold:
                break
        return history[:epoch]

    # make prediction
    def predict(self, src, device):
        """ Predict function for a trained model to predict
        """
        self.eval()
        src = torch.as_tensor(src).float()
        src = src.to(device)

        return self.forward(src, device).detach().cpu().numpy()

# There are two types of Encoder-FNN AR model
# (1) Read orginial AR seq
# (2) Include all climate index


class EncoderFNN_AllSeq_AR(nn.Module):
    """The class for an encoder (LSTM)-decoder(FNN) model where the input of the decoder is the output of all steps in input sequence
    plus an Autoregressive module using historical sequence as input
    """
    def __init__(self, input_dim, output_dim, hidden_dim=10, num_layers=2, seq_len=4, linear_dim=100, learning_rate=0.01, dropout=0.1, threshold=0.1, num_epochs=100):
        """ Initilize an encoder (LSTM)-decoder(FNN) model where the input of the decoder is the output of all steps in input sequence
        plus an Autoregressive module using historical sequence as input
        Args:
        input_dim: int -- the size/dimensionality of the vectors that will be input for encoder
        output_dim: int -- the size/dimensionality of the vectors that will be input for decoder
        hidden_dim: int -- the dimensionality of the hidden and cell states
        num_layers: int -- the number of layers in the LSTM
        seq_len: int -- the length of input sequence for AR module and encoder
        linear_dim: int -- the dimensionality of the decoder FNN
        learning_rate: float -- learning learning_rate
        threshold: float -- the early stopping point (training loss < threshold) for training process
        num_epochs: int -- the maximum number of training epochs
        """
        super().__init__()

        self.input_dim = input_dim  # batch_size seq_length n_features
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.threshold = threshold
        self.seq_len = seq_len
        self.dropout = dropout
        self.linear_dim = linear_dim
        self.weight_decay = 0.0

        self.encoder = Encoder_all_seq(self.input_dim, self.hidden_dim, self.num_layers, self.dropout)

        self.out1 = nn.Linear(self.hidden_dim * self.seq_len, self.linear_dim)
        self.out2 = nn.Linear(self.linear_dim, self.output_dim)
        self.ar_dim = seq_len
        self.ar = Rowwise_mult_layer(self.ar_dim, self.output_dim)
        self.comb = nn.Linear(2, 1)

        # assert self.encoder.hidden_dim == self.decoder.hidden_dim, \
        #     "Hidden dimensions of encoder and decoder must be equal!"
        # assert self.encoder.num_layers == self.decoder.num_layers, \
        #     "Encoder and decoder must have equal number of layers!"

    def forward(self, src, target_ar, device):
        """Forward function
        """
        # src = [batch size,sent len, src] e.g.[5,10,1]
        # trg = [batch size,sent len, trg] e.g.[5,10,1]

        src = torch.as_tensor(src[:, -self.seq_len:, :]).float()
        src = src.to(device)
        target_ar = torch.as_tensor(target_ar[:, :, -self.seq_len:]).float()
        target_ar = target_ar.to(device)
        hidden_ar = self.ar(target_ar)
        batch_size = src.shape[0]  # batch_size
        seq_len = src.shape[1]

        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size,self.decoder_len,self.output_dim).to(device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        encoder_output = self.encoder(src)

        # encoder_output = encoder_output.permute(1,0,2)

        encoder_output = encoder_output.reshape(encoder_output.shape[0], encoder_output.shape[1] * encoder_output.shape[2])

        linear_output = self.out1(encoder_output)
        linear_output = F.relu(linear_output)
        output = self.out2(linear_output)

        result = self.comb(torch.cat((hidden_ar.unsqueeze(2), output.unsqueeze(2)), axis=2))
        return result.squeeze(2)

    def fit(self, train_loader, device):
        """ Fit function for model training
        """
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = torch.nn.MSELoss(reduction='mean')  # sum of the error for all element in the batch
        max_epoch = self.num_epochs

        for epoch in range(max_epoch):

            self.train()
            train_epoch_loss = 0
            for j, (src, trg_ar, trg) in enumerate(train_loader):

                trg = torch.as_tensor(trg).float()
                trg = trg.to(device)
                if len(src.size()) < 3:
                    src = src.view(1, -1)
                trg_ar = torch.as_tensor(trg_ar).float()
                train_output = self.forward(src, trg_ar, device)
#                print(torch.isnan(train_output).any())
                loss = criterion(train_output, trg)
#                print(loss)
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_epoch_loss += loss.item()

            print('Epoch: {}/{} Train Loss: {:.4f}'.format(epoch, max_epoch, train_epoch_loss / (j + 1)))

            if train_epoch_loss / (j + 1) < self.threshold:
                break

    def fit_cv(self, train_loader, val_src, val_trg_ar, val_trg, device):
        """ Fit function for hyper-parameter tuning
        """
        val_src = torch.as_tensor(val_src).float()
        val_trg = torch.as_tensor(val_trg).float()
        val_trg_ar = torch.as_tensor(val_trg_ar).float()
        val_src = val_src.to(device)
        val_trg = val_trg.to(device)
        val_trg_ar = val_trg_ar.to(device)

        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = torch.nn.MSELoss(reduction='mean')  # sum of the error for all element in the batch
        history = np.zeros((self.num_epochs, 2))

        for epoch in range(self.num_epochs):
            self.train()
            train_epoch_loss = 0
            for i, (src, trg_ar, trg) in enumerate(train_loader):
                src = torch.as_tensor(src).float()
                trg_ar = torch.as_tensor(trg_ar).float()
                train_output = self.forward(src, trg_ar, device)
                trg = torch.as_tensor(trg).float()
                trg = trg.to(device)
                loss = criterion(train_output, trg)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_epoch_loss += loss.item()
            # on validation set
            self.eval()
            val_output = self.forward(val_src, val_trg_ar, device)
            loss = criterion(val_output, val_trg)
            val_epoch_loss = loss.item()
            history[epoch] = [train_epoch_loss / (i + 1), val_epoch_loss]
            print('Epoch: {}/{} Train Loss: {:.4f} Validation Loss:{:.4f}'.format(epoch, self.num_epochs, train_epoch_loss / (i + 1), val_epoch_loss))
            if train_epoch_loss / (i + 1) < self.threshold:
                break
        return history[:epoch]

        # make prediction
    def predict(self, src, target_ar, device):
        """ Predict function for a trained model to predict
        """
        self.eval()
        src = torch.as_tensor(src).float()
        target_ar = torch.as_tensor(target_ar).float()
        src = src.to(device)
        target_ar = target_ar.to(device)
        return self.forward(src, target_ar, device).detach().cpu().numpy()


class EncoderFNN_AllSeq_AR_CI(nn.Module):
    """The class for an encoder (LSTM)-decoder(FNN) model where the input of the decoder is the output of all steps in input sequence
    plus an Autoregressive module using historical sequence and climate indexes (temporal variables) as input
    """
    def __init__(self, input_dim, output_dim, hidden_dim=10, num_layers=2, seq_len=4, linear_dim=100, ci_dim=8, learning_rate=0.01, dropout=0.1, threshold=0.1, num_epochs=100):
        """ Initilize an encoder (LSTM)-decoder(FNN) model where the input of the decoder is the output of all steps in input sequence
        plus an Autoregressive module using historical sequence and climate indexes (temporal variables) as input
        Args:
        input_dim: int -- the size/dimensionality of the vectors that will be input for encoder
        output_dim: int -- the size/dimensionality of the vectors that will be input for decoder
        hidden_dim: int -- the dimensionality of the hidden and cell states
        num_layers: int -- the number of layers in the LSTM
        seq_len: int -- the length of input sequence for AR module and encoder
        linear_dim: int -- the dimensionality of the decoder FNN
        ci_dim: int -- the number of climate indexes (the dimensionality of temporal variables)
        learning_rate: float -- learning learning_rate
        threshold: float -- the early stopping point (training loss < threshold) for training process
        num_epochs: int -- the maximum number of training epochs
        """
        super().__init__()

        self.input_dim = input_dim  # batch_size seq_length n_features
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.threshold = threshold
        self.seq_len = seq_len
        self.dropout = dropout
        self.linear_dim = linear_dim
        self.weight_decay = 0.0

        self.encoder = Encoder_all_seq(self.input_dim, self.hidden_dim, self.num_layers, self.dropout)

        self.out1 = nn.Linear(self.hidden_dim * self.seq_len, self.linear_dim)
        self.out2 = nn.Linear(self.linear_dim, self.output_dim)
        self.ar_dim = seq_len + ci_dim
        self.ar = Rowwise_mult_layer(self.ar_dim, self.output_dim)
        self.comb = nn.Linear(2, 1)

        # assert self.encoder.hidden_dim == self.decoder.hidden_dim, \
        #     "Hidden dimensions of encoder and decoder must be equal!"
        # assert self.encoder.num_layers == self.decoder.num_layers, \
        #     "Encoder and decoder must have equal number of layers!"

    def forward(self, src, target_ar, device):
        """Forward function
        """
        # src = [batch size,sent len, src] e.g.[5,10,1]
        # trg = [batch size,sent len, trg] e.g.[5,10,1]

        climate_index = torch.as_tensor(src[:, -1, self.input_dim:]).float()
        src = torch.as_tensor(src[:, -self.seq_len:, :self.input_dim]).float()
        # src = torch.as_tensor(src).float()
        src = src.to(device)
        climate_index = climate_index.to(device)
        climate_index = climate_index.unsqueeze(1)
        climate_index = climate_index.repeat(1, self.output_dim, 1)
        target_ar = torch.as_tensor(target_ar[:, :, -self.seq_len:]).float()
        target_ar = target_ar.to(device)
        hidden_ar = self.ar(torch.cat((target_ar, climate_index), axis=2))
        batch_size = src.shape[0]  # batch_size
        seq_len = src.shape[1]

        # tensor to store decoder outputs
        # outputs = torch.zeros(batch_size,self.decoder_len,self.output_dim).to(device)

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        encoder_output = self.encoder(src)

        # encoder_output = encoder_output.permute(1,0,2)

        encoder_output = encoder_output.reshape(encoder_output.shape[0], encoder_output.shape[1] * encoder_output.shape[2])

        linear_output = self.out1(encoder_output)
        linear_output = F.relu(linear_output)
        output = self.out2(linear_output)

        result = self.comb(torch.cat((hidden_ar.unsqueeze(2), output.unsqueeze(2)), axis=2))
        return result.squeeze(2)

    def fit(self, train_loader, device):
        """ Fit function for model training
        """
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = torch.nn.MSELoss(reduction='mean')  # sum of the error for all element in the batch
        max_epoch = self.num_epochs

        for epoch in range(max_epoch):

            self.train()
            train_epoch_loss = 0
            for j, (src, trg_ar, trg) in enumerate(train_loader):

                trg = torch.as_tensor(trg).float()
                trg = trg.to(device)
                if len(src.size()) < 3:
                    src = src.view(1, -1)
                trg_ar = torch.as_tensor(trg_ar).float()
                train_output = self.forward(src, trg_ar, device)
#                print(torch.isnan(train_output).any())
                loss = criterion(train_output, trg)
#                print(loss)
                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_epoch_loss += loss.item()

            print('Epoch: {}/{} Train Loss: {:.4f}'.format(epoch, max_epoch, train_epoch_loss / (j + 1)))

            if train_epoch_loss / (j + 1) < self.threshold:
                break

    def fit_cv(self, train_loader, val_src, val_trg_ar, val_trg, device):
        """ Fit function for hyper-parameter tuning
        """
        val_src = torch.as_tensor(val_src).float()
        val_trg = torch.as_tensor(val_trg).float()
        val_trg_ar = torch.as_tensor(val_trg_ar).float()
        val_src = val_src.to(device)
        val_trg = val_trg.to(device)
        val_trg_ar = val_trg_ar.to(device)

        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = torch.nn.MSELoss(reduction='mean')  # sum of the error for all element in the batch
        history = np.zeros((self.num_epochs, 2))

        for epoch in range(self.num_epochs):
            self.train()
            train_epoch_loss = 0
            for i, (src, trg_ar, trg) in enumerate(train_loader):
                src = torch.as_tensor(src).float()
                trg_ar = torch.as_tensor(trg_ar).float()
                train_output = self.forward(src, trg_ar, device)
                trg = torch.as_tensor(trg).float()
                trg = trg.to(device)
                loss = criterion(train_output, trg)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_epoch_loss += loss.item()
            # on validation set
            self.eval()
            val_output = self.forward(val_src, val_trg_ar, device)
            loss = criterion(val_output, val_trg)
            val_epoch_loss = loss.item()
            history[epoch] = [train_epoch_loss / (i + 1), val_epoch_loss]
            print('Epoch: {}/{} Train Loss: {:.4f} Validation Loss:{:.4f}'.format(epoch, self.num_epochs, train_epoch_loss / (i + 1), val_epoch_loss))
            if train_epoch_loss / (i + 1) < self.threshold:
                break
        return history[:epoch]

        # make prediction
    def predict(self, src, target_ar, device):
        """ Predict function for a trained model to predict
        """
        self.eval()
        src = torch.as_tensor(src).float()
        target_ar = torch.as_tensor(target_ar).float()
        src = src.to(device)
        target_ar = target_ar.to(device)
        return self.forward(src, target_ar, device).detach().cpu().numpy()
