# TranslationService

TranslationService is a service for translating one language into another. Can translate single words or paragraphs.

## Technologies

Below is a list of all the technologies used to develop the app. They are structured by categories.

**Backend**
   - Python
   - Flask
   - PyTorch

## Run & Build commands

The application is developed using the python.

Before executing any command, we must make sure that the packages are installed, otherwise we must install them using this command:
```
pip install -r requirements.txt
```

To run the application in the development mode use this command:
```
python app.py
```

## Management of Folder & Files

The project is structured as follows:

- [Data](https://github.com/EddyEduard/TranslationService/tree/main/data) used for storing word and paragraph files in two different languages;
- [Models](https://github.com/EddyEduard/TranslationService/tree/main/models) are used for storing the trained models;
- [Utils](https://github.com/EddyEduard/TranslationService/tree/main/utils) containing some useful functions.

## Implementation

The service was implemented using neural networks. Since translation involves predicting the next word in the sentence, traditional neural networks cannot solve this problem. Therefore, over time, different models have been implemented to solve this translation problem and one of the most popular is **Seq2Seq** (sequence-to-sequence).

The architecture of a **Seq2Seq** model is composed of two main parts:

### Encoder

The encoder processes the input sequence and compresses the information into a context vector (or a set of vectors). Typically, the encoder is a Recurrent Neural Network (RNN), Long Short-Term Memory (LSTM), or Gated Recurrent Unit (GRU) that reads the input sequence one token at a time and updates its hidden state accordingly.
   
### Decoder

The decoder generates the output sequence from the context vector provided by the encoder. Like the encoder, the decoder is also an RNN, LSTM, or GRU. It uses the context vector as the initial hidden state and generates the output sequence step-by-step. At each step, the decoder predicts the next token in the sequence based on its current hidden state and the previous token generated.

The current application has been trained to translate paragraphs with maximum 7 words from **English** to **Romanian**.

## License
Distributed under the MIT License. See [MIT](https://github.com/EddyEduard/TranslationService/blob/master/LICENSE) for more information.

## Contact
EddyEduard - [eduard_nicolae@yahoo.com](mailTo:eduard_nicolae@yahoo.com)
\
Project link - [https://github.com/EddyEduard/TranslationService](https://github.com/EddyEduard/TranslationService.git)
