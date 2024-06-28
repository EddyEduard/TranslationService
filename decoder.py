import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''

Decoder

- this class takes the output data and decodes it based on the hidden data and context provided by the encoder
- the decoder is formatted from layers:
    1. Embedding
    2. LSTM
    3. Linear
    4. LogSoftmax

'''


class DecoderRNN(nn.Module):

    '''

    Initialization encoder

    hidden_size - hidden units count
    output_size - output words count

    '''

    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    '''
    
    Perform forward propagation through decoder

    input - the input sample at a given point in time
    hidden - the hidden state from preview cell
    context - the context state from preview cell

    Return the output, hidden and context state from current cell

    '''

    def forward(self, input, hidden, context):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, (hidden, context) = self.lstm(output, (hidden, context))
        output = self.softmax(self.linear(output[0]))

        return output, (hidden, context)

    # Initialize the hidden state for first cell.
    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

    # Initialize the context state for first cell.
    def init_context(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


'''

Decoder with attention

- this class takes the output data and decodes it based on the hidden data and context provided by the encoder
- the attention decoder is formatted from layers:
    1. Embedding
    2. Linear
    3. Linear
    4. Dropout
    5. LSTM
    6. Linear

'''


class AttentionDecoderRNN(nn.Module):

    '''

    Initialization attention decoder

    hidden_size - hidden units count
    output_size - output words count
    dropout_p - probability of an element to be zeroed
    max_length - maximum sentence length

    '''

    def __init__(self, hidden_size, output_size, dropout_p, max_length):
        super(AttentionDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attention = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attention_combine = nn.Linear(
            self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.lstm = nn.LSTM(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    '''
    
    Perform forward propagation through attention decoder

    input - the input sample at a given point in time
    hidden - the hidden state from preview cell
    context - the context state from preview cell
    encoder_outputs - the output from encoder cell

    Return the output, hidden, context and attention weights state from current cell

    '''

    def forward(self, input, hidden, context, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attention_weights = F.softmax(
            self.attention(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attention_applied = torch.bmm(attention_weights.unsqueeze(0),
                                      encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attention_applied[0]), 1)
        output = self.attention_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, (hidden, context) = self.lstm(output, (hidden, context))

        output = F.log_softmax(self.out(output[0]), dim=1)

        return output, (hidden, context), attention_weights

    # Initialize the hidden state for first cell.
    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

    # Initialize the context state for first cell.
    def init_context(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)
