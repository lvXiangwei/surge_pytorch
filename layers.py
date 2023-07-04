import torch
import torch.nn as nn

class AUGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, bias = True):
        super(AUGRUCell, self).__init__()

        in_dim = input_dim + hidden_dim
        self.reset_gate = nn.Sequential( nn.Linear( in_dim, hidden_dim, bias = bias), nn.Sigmoid())
        self.update_gate = nn.Sequential( nn.Linear( in_dim, hidden_dim, bias = bias), nn.Sigmoid())
        self.h_hat_gate = nn.Sequential( nn.Linear( in_dim, hidden_dim, bias = bias), nn.Tanh())


    def forward(self, X, h_prev, attention_score):
        temp_input = torch.cat( [ h_prev, X ] , dim = -1)
        r = self.reset_gate( temp_input)
        u = self.update_gate( temp_input)

        h_hat = self.h_hat_gate( torch.cat( [ h_prev * r, X], dim = -1) )

        u = attention_score.unsqueeze(1) * u
        h_cur = (1. - u) * h_prev + u * h_hat

        return h_cur


class DynamicGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.rnn_cell = AUGRUCell( input_dim, hidden_dim, bias = True)

    def forward(self, X, attenion_scores , h0 = None ):
        B, T, D = X.shape
        H = self.hidden_dim
        
        output = torch.zeros( B, T, H ).type( X.type() )
        h_prev = torch.zeros( B, H ).type( X.type() ) if h0 == None else h0
        for t in range( T): 
            h_prev = output[ : , t, :] = self.rnn_cell( X[ : , t, :], h_prev, attenion_scores[ :, t] )
        return output
    

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, device="cuda"):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Output layer
        # self.fc = nn.Linear(hidden_size, output_size)

        self.device = device
    
    def forward(self, x, seq_len):
        # Initialize hidden state and cell state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(self.device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  
        
        # Index into the last element of each sequence
        idx = (seq_len-1).view(-1, 1).unsqueeze(-1).expand(len(seq_len), 1, self.hidden_size)
        out = out.gather(1, idx).squeeze(1)
        
        # Decode the hidden state of the last tnime step
        # out = self.fc(out) 
        
        return out