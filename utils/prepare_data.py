from io import open

import unicodedata
import re
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SOS_TOKEN = 0
EOS_TOKEN = 1

'''

Transform sentences

'''

class SentenceTransform:

    '''
    
    Initialization

    name - name transformations
    
    '''

    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.count_words = 2

    '''
    
    Add a word

    word - word to add
    
    '''

    def __add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.count_words
            self.word2count[word] = 1
            self.index2word[self.count_words] = word
            self.count_words += 1
        else:
            self.word2count[word] += 1

    '''
    
    Add a sentence

    sentence - sentence to add
    
    '''

    def add_sentence(self, sentence):
        for word in sentence.split(' '):
            self.__add_word(word)

    '''
    
    Provide indexes from a sentence

    language - the language from which to extract clues for the sentence
    sentence - the sentence which to be indexed

    Return the indexed sentence
    
    '''

    def indexes_from_sentence(self, language, sentence):
        return [language.word2index[word] for word in sentence.split(' ')]

    '''
    
    Tokenize a sentence

    sentence - the sentence which to be tokenized
    
    Return the tokenized sentence

    '''

    def tensor_from_sentence(self, sentence):
        indexes = self.indexes_from_sentence(self, sentence)
        indexes.append(EOS_TOKEN)
        return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


'''

Prepare data from a file. Extract it, normalize it and process it

'''

class PrepareData:

    '''
    
    Initialization

    first_language - first language name
    second_language - second language name
    datasets_path - datasets path
    max_length - maximum sentence length
    
    '''

    def __init__(self, first_language, second_language, datasets_path, max_length):
        self.first_language = first_language
        self.second_language = second_language
        self.datasets_path = datasets_path
        self.max_length = max_length

    '''
    
    Encoding the sentence in ASCII

    sentence - the sentence which to be encoding
    
    Return the encoded sentence

    '''

    def __unicode_sentence_to_ascii(self, sentence):
        return ''.join(
            s for s in unicodedata.normalize("NFD", sentence)
            if unicodedata.category(s) != "Mn"
        )
    
    '''
    
    Sentence normalization

    sentence - the sentence to be normalized

    Return the normalized sentence
    
    '''

    def __normalize_sentence(self, sentence):
        sentence = self.__unicode_sentence_to_ascii(sentence.lower().strip())
        sentence = re.sub(r"([.!?])", r" \1", sentence)
        sentence = re.sub(r"[^a-zA-Z.!?]+", r" ", sentence)

        return sentence.strip()

    '''
    
    Read the data from file and prepare it for the transformation

    Return the transformed source sentence, transformed target sentence, and sentence pairs
    
    '''

    def __read_source_and_target_languages(self):
        lines = open(self.datasets_path,
                     encoding='utf-8').read().strip().split('\n')

        sentence_pairs = [[self.__normalize_sentence(sentence) for sentence in line.split('\t')]
                          for line in lines if len(line.split('\t')[0].split(" ")) == self.max_length]

        source_sentence = SentenceTransform(self.first_language)
        target_sentence = SentenceTransform(self.second_language)

        return source_sentence, target_sentence, sentence_pairs

    '''
    
    Process the data
    
    Return the source sentences, target senteces and sentence pairs

    '''

    def process(self):
        source_sentence, target_sentence, sentence_pairs = self.__read_source_and_target_languages()

        for pair in sentence_pairs:
            source_sentence.add_sentence(pair[0])
            target_sentence.add_sentence(pair[1])

        return source_sentence, target_sentence, sentence_pairs
