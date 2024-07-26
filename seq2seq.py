import os
import time
import math
import random
import torch
import torch.nn as nn

from __future__ import unicode_literals, print_function, division
from torch import optim
from encoder import EncoderRNN
from decoder import AttentionDecoderRNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_TOKEN = 0
EOS_TOKEN = 1

'''

Sequence to Sequence

- this class takes input data, encodes it, and then tries to decode it by predicting the next word in the sentence
- the seq2seq is formatted from:
    1. Encoder
    2. Decoder

'''

class Seq2Seq:

    '''

    Initialization seq2seq

    interation - number of epochs
    learning_rate - learning rate
    hidden_size - hidden units count
    max_length - maximum sentence length
    dropout_p - probability of an element to be zeroed
    teacher_forcing_ratio - teacher forcing ratio rate
    print_every - show the status of the algorithm for each number of completed epochs

    '''

    def __init__(self, interation=10000, learning_rate=0.01, hidden_size=256, max_length=10, dropout_p=0.5, teacher_forcing_ratio=0.5, print_every=1000):
        self.interation = interation
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.max_length = max_length
        self.dropout_p = dropout_p
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.print_every = print_every


    '''
    
    Fit the data used for training

    source_sentence - the input data which contains the tokenized words from source sentence 
    target_sentence - the output data which contains the tokenized words from target sentence 
    sentence_pairs - pairs between the source sentence and the target sentence
    
    '''

    def fit(self, source_sentence, target_sentence, sentence_pairs):
        self.source_sentence = source_sentence
        self.target_sentence = target_sentence
        self.sentence_pairs = sentence_pairs

    '''
    
    Prepare the alghoritm for training
    
    '''

    def prepare(self):
        self.encoder = EncoderRNN(
            self.source_sentence.count_words, self.hidden_size).to(device)

        self.decoder = AttentionDecoderRNN(
            self.hidden_size, self.target_sentence.count_words, dropout_p=self.dropout_p, max_length=self.max_length).to(device)

        # self.decoder = DecoderRNN(
        #     self.hidden_size, self.target_sentence.count_words).to(device)

    '''
    
    Choose to save the trained model state with a specific name

    name_state_dict - the name from the trained model state

    '''

    def save(self, name_state_dict=None):
        self.name_state_dict = name_state_dict

    '''
    
    Choose to test a trained model state

    name_state_dict - the name from the trained model state

    '''

    def test(self, name_state_dict):
        self.encoder.load_state_dict(
            torch.load("models/" + name_state_dict + "/encoder-state.dict"))
        self.decoder.load_state_dict(
            torch.load("models/" + name_state_dict + "/decoder-state.dict"))

    def __as_minutes(self, s):
        m = math.floor(s / 60)
        s -= m * 60
        return '%dm %ds' % (m, s)

    def __time_since(self, since, percent):
        now = time.time()
        s = now - since
        es = s / (percent)
        rs = es - s
        return '%s (- %s)' % (self.__as_minutes(s), self.__as_minutes(rs))
    
    '''
    
    Perform a single interation from training

    source_tensor - input sentence
    terget_tensor - output sentence 
    encoder_optimizer - optimizer for encoder
    decoder_optimizer - optimizer for decoder
    criterion - the loss error calculation criterion
    
    Return the loss error value

    '''

    def __train_interation(self, source_tensor, target_tensor, encoder_optimizer, decoder_optimizer, criterion):
        loss = 0

        source_length = source_tensor.size(0)
        target_length = target_tensor.size(0)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        # Perform forward propagation through encoder for a sentence

        encoder_hidden = self.encoder.init_hidden() 
        encoder_context = self.encoder.init_context()

        encoder_outputs = torch.zeros(
            self.max_length, self.encoder.hidden_size, device=device)

        for i in range(source_length):
            encoder_output, (encoder_hidden, encoder_context) = self.encoder(
                source_tensor[i], encoder_hidden, encoder_context)
            encoder_outputs[i] = encoder_output[0, 0]

        # Perform forward propagation through decoder for a encoded sentence

        decoder_input = torch.tensor([[SOS_TOKEN]], device=device)

        decoder_hidden = encoder_hidden
        decoder_context = encoder_context

        use_teacher_forcing = True if random.random(
        ) < self.teacher_forcing_ratio else False

        if use_teacher_forcing:
            for i in range(target_length):
                decoder_output, (decoder_hidden, decoder_context), _ = self.decoder(
                    decoder_input, decoder_hidden, decoder_context, encoder_outputs)
                decoder_input = target_tensor[i]
                loss += criterion(decoder_output, target_tensor[i])
        else:
            for i in range(target_length):
                decoder_output, (decoder_hidden, decoder_context), _ = self.decoder(
                    decoder_input, decoder_hidden, decoder_context, encoder_outputs)
                decoder_input = torch.argmax(decoder_output)
                loss += criterion(decoder_output, target_tensor[i])

                if decoder_input.item() == EOS_TOKEN:
                    break

        # Backpropagate through the algorithm starting at the decoder and ending at the encoder

        loss.backward()

        # Update the parameters in the encoder

        encoder_optimizer.step()

        # Update the parameters in the decoder

        decoder_optimizer.step()

        return loss.item() / target_length

    '''
    
    Training the algorithm
    
    '''

    def train(self):
        start = time.time()
        loss_total = 0

        # Initialization of the stochastic gradient descent for the encoder

        encoder_optimizer = optim.SGD(
            self.encoder.parameters(), lr=self.learning_rate)
        
        # Initialization of the stochastic gradient descent for the decoder

        decoder_optimizer = optim.SGD(
            self.decoder.parameters(), lr=self.learning_rate)

        # Initialize the criterion used to calculate the loss error

        criterion = nn.NLLLoss()

        # Choose a random pair of sentences to start training

        random_pairs = [random.choice(self.sentence_pairs)
                        for i in range(self.interation)]
        
        # Perform each interation

        for iteration in range(1, self.interation + 1):

            # Tokenization of the source sentence

            source_tensor = self.source_sentence.tensor_from_sentence(
                random_pairs[iteration - 1][0])
            
            # Tokenization of the terget sentence

            target_tensor = self.target_sentence.tensor_from_sentence(
                random_pairs[iteration - 1][1])

            # Perform a single interation and sum it to the total loss error

            loss_total += self.__train_interation(
                source_tensor, target_tensor, encoder_optimizer, decoder_optimizer, criterion)

            # Show at the current iteration the state of the algorithm

            if iteration % self.print_every == 0:
                print_loss_avg = loss_total / self.print_every
                loss_total = 0

                print('%s (%d %d%%) %.4f' % (self.__time_since(
                    start, iteration / self.interation), iteration, iteration / self.interation * 100, print_loss_avg))

        # Save the state of the trained model after training is complete

        if self.name_state_dict != None:
            if not os.path.exists("models/" + self.name_state_dict):
                os.makedirs("models/" + self.name_state_dict)

            torch.save(self.encoder.state_dict(), "models/" +
                       self.name_state_dict + "/encoder-state.dict")
            torch.save(self.decoder.state_dict(), "models/" +
                       self.name_state_dict + "/decoder-state.dict")

    '''
    
    Predict a sentence for a trained model

    sentence - input sentence

    Return the output sentence
    
    '''

    def predict(self, sentence):
        try:
            with torch.no_grad():

                # Tokenization of the sentence

                source_tensor = self.source_sentence.tensor_from_sentence(
                    sentence)
                
                source_length = source_tensor.size()[0]

                # Perform forward propagation through encoder for a sentence

                encoder_hidden = self.encoder.init_hidden()
                encoder_context = self.encoder.init_context()

                encoder_outputs = torch.zeros(
                    self.max_length, self.encoder.hidden_size, device=device)

                for i in range(source_length):
                    encoder_output, (encoder_hidden, encoder_context) = self.encoder(
                        source_tensor[i], encoder_hidden, encoder_context)
                    encoder_outputs[i] += encoder_output[0, 0]

                # Perform forward propagation through decoder for a encoded sentence

                decoder_input = torch.tensor([[SOS_TOKEN]], device=device)

                decoder_hidden = encoder_hidden
                decoder_context = encoder_context

                decoded_words = []
                decoder_attentions = torch.zeros(
                    self.max_length, self.max_length)

                for i in range(self.max_length):
                    decoder_output, (decoder_hidden, decoder_context), decoder_attention = self.decoder(
                        decoder_input, decoder_hidden, decoder_context, encoder_outputs)
                    decoder_attentions[i] = decoder_attention.data
                    decoder_input = torch.argmax(decoder_output)

                    if decoder_input.item() == EOS_TOKEN:
                        break
                    else:
                        decoded_words.append(
                            self.target_sentence.index2word[decoder_input.item()])

                return " ".join(decoded_words)
        except Exception as e:
            print(e)
