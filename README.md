# TranslationService

TranslationService is a service for translating one language into another. Can translate single words or paragraphs.

## Technologies

Below is a list of all the technologies used to develop the app. They are structured by categories.

**Backend**
   - Python
   - Flask
   - PyTorch

## Environment variables

Before running the application, is necessary to set the environment variables used by the application. For this, a file named **.env** must be created. This file will contain all environment variables.

The environment variables that are used by the application are:

```
HOST = "0.0.0.0"
```

The **HOST** variable contains the IP address where the application is running.

```
PORT = 7000
```

The **PORT** variable contains the port where the application is running.

```
TS_MODEL_NAME = "<<your model name>>"
```

The **TS_MODEL_NAME** variable contains the name of the model.

```
TS_FIRST_LANGUAGE = "English"
```

The **TS_FIRST_LANGUAGE** variable contains the name of the first language.

```
TS_SECOND_LANGUAGE = "Romanian"
```

The **TS_SECOND_LANGUAGE** variable contains the name of the second language.

```
TS_DATASETS_PATH = "data/en-ro.txt"
```

The **TS_DATASETS_PATH** variable contains the datasets path.

```
TS_COUNT_WORDS = 12
```

The **TS_COUNT_WORDS** variable contains the count of words.

```
TS_MAX_LENGTH = 50
```

The **TS_MAX_LENGTH** variable contains the max length of statement.

```
TS_INTERATION = 1000000
```

The **TS_INTERATION** variable contains the number of interations.

```
TS_LEARNING_RATE = 0.001
```

The **TS_LEARNING_RATE** variable contains the learning rate.

```
TS_HIDDEN_SIZE = 256
```

The **TS_HIDDEN_SIZE** variable contains the number of hidden nodes.

```
TS_DROPOUT_P = 0.5
```

The **TS_DROPOUT_P** variable contains the dropout value.

```
TS_TEACHER_FORCING_RATIO = 0.5
```

The **TS_TEACHER_FORCING_RATIO** variable contains the teacher forcing ratio.

```
TS_PRINT_EVERY = 1000
```

The **TS_PRINT_EVERY** variable contains the value to print the state of the training model after each interacted number.

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

## Contarization

For contarization the service was used [Docker](https://www.docker.com/). It is recommended to use Docker containers for training a new model. 

To create a new Docker image for this service, run the following command:
```
docker build . -t translation-service:1.0
```

After finishing training a model, it is necessary to save the states (coefficients, parameters, etc.) of the model in a certain place. Since it is possible to have many containers created from the same Docker image, it would be very difficult to retrieve every state of the model after it has been trained. Thus, we can create a volume to store all the states of the trained model. 

To create a volume for storing models, run the following command:
```
docker volume create translation-service-models
```

After creating the image and volume, the next step is to create the containers. Each container can be specialized to train a different model. The only thing that needs to be done is to set the environment variables that specify the parameters used to train the model.

To create a container, run the following command:
```
docker run -itd --name translation-service-12-words -e TS_MODEL_NAME="model-12" -e TS_FIRST_LANGUAGE="English" -e TS_SECOND_LANGUAGE="Romanian" -e TS_DATASETS_PATH="data/en-ro.txt" -e TS_COUNT_WORDS=12 -e TS_MAX_LENGTH=50 -e TS_INTERATION=1000000 -e TS_LEARNING_RATE=0.001 -v translation-service-models:/TranslationService/models translation-service:1.0
```

Finally, the following command must be executed to start model training:
```
docker exec -it [CONTAINER ID] pm2 start app.py --no-autorestart
```

## License
Distributed under the MIT License. See [MIT](https://github.com/EddyEduard/TranslationService/blob/master/LICENSE) for more information.

## Contact
EddyEduard - [eduard_nicolae@yahoo.com](mailTo:eduard_nicolae@yahoo.com)
\
Project link - [https://github.com/EddyEduard/TranslationService](https://github.com/EddyEduard/TranslationService.git)
