import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''

Encoder

- this class takes the input data and encodes it
- the encoder is formatted from layers:
    1. Embedding
    2. LSTM

'''


class EncoderRNN(nn.Module):

    '''

    Initialization encoder

    input_size - input words count
    hidden_size - hidden units count

    '''

    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)

    '''
    
    Perform forward propagation through encoder

    input - the input sample at a given point in time
    hidden - the hidden state from preview cell
    context - the context state from preview cell

    Return the output, hidden and context state from current cell

    '''

    def forward(self, input, hidden, context):
        embedded = self.embedding(input).view(1, 1, -1)
        output, (hidden, context) = self.lstm(embedded, (hidden, context))

        return output, (hidden, context)

    # Initialize the hidden state for first cell.
    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

    # Initialize the context state for first cell.
    def init_context(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
