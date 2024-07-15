import os
import re

from os import environ
from dotenv import load_dotenv
from utils.prepare_data import PrepareData
from seq2seq import Seq2Seq
from flask import Flask, request

# Load environment variables.

load_dotenv()

# Init Flask app.

app = Flask(__name__)

'''

Train the model.

name - name of trained model
first_language - first language name
second_language - second language name
datasets_path - datasets path
count_words - number of the words
max_length - maximum sentence length
interation - number of epochs
learning_rate - learning rate
hidden_size - hidden units count
max_length - maximum sentence length
dropout_p - probability of an element to be zeroed
teacher_forcing_ratio - teacher forcing ratio rate
print_every - show the status of the algorithm for each number of completed epochs

'''

def train_model(name, first_language, second_language, datasets_path, count_words, max_length, interation, learning_rate, hidden_size, dropout_p, teacher_forcing_ratio, print_every):
    data = PrepareData(first_language, second_language, datasets_path, count_words)
    source_sentence, target_sentence, sentence_pairs = data.process()

    seq2seq = Seq2Seq(interation=interation,
                      learning_rate=learning_rate,
                      hidden_size=hidden_size,
                      max_length=max_length,
                      dropout_p=dropout_p,
                      teacher_forcing_ratio=teacher_forcing_ratio,
                      print_every=print_every
                      )
    seq2seq.fit(source_sentence, target_sentence, sentence_pairs)
    seq2seq.prepare()
    seq2seq.save(name)
    seq2seq.train()

'''

Prepare a trained model for testing.

name - name of trained model
first_language - first language name
second_language - second language name
datasets_path - datasets path
count_words - number of the words
max_length - maximum sentence length

'''

def prepare_model(name, first_language, second_language, datasets_path, count_words, max_length):
    data = PrepareData(first_language, second_language, datasets_path, count_words)
    source_sentence, target_sentence, sentence_pairs = data.process()

    seq2seq = Seq2Seq(hidden_size=256, max_length=max_length, dropout_p=0.1)
    seq2seq.fit(source_sentence, target_sentence, sentence_pairs)
    seq2seq.prepare()
    seq2seq.test(name)

    return seq2seq

'''

Translate a text.

text - text for translation
first_language - first language name
second_language - second language name
datasets_path - datasets path

'''

def translate(text, first_language, second_language, datasets_path):
    if re.search(r"([.!?])", text):
        text_split = re.split(r"([.!?])", text)
        translate = ""

        for i in range(0, len(text_split)):
            if re.search(r"([.!?])", text_split[i]):
                sentence = "{0} {1}".format(
                    text_split[i - 1].lower(),  text_split[i])

                if len(sentence.split(" ")) == 2:
                    model = prepare_model("model-1", first_language, second_language, datasets_path, 1, 5)
                elif len(sentence.split(" ")) == 3:
                    model = prepare_model("model-2", first_language, second_language, datasets_path, 2, 10)
                elif len(sentence.split(" ")) == 4:
                    model = prepare_model("model-3", first_language, second_language, datasets_path, 3, 10)
                elif len(sentence.split(" ")) == 5:
                    model = prepare_model("model-4", first_language, second_language, datasets_path, 4, 15)
                elif len(sentence.split(" ")) == 6:
                    model = prepare_model("model-5", first_language, second_language, datasets_path, 5, 20)
                elif len(sentence.split(" ")) == 7:
                    model = prepare_model("model-6", first_language, second_language, datasets_path, 6, 25)
                elif len(sentence.split(" ")) == 8:
                    model = prepare_model("model-7", first_language, second_language, datasets_path, 7, 30)
                elif len(sentence.split(" ")) == 9:
                    model = prepare_model("model-8", first_language, second_language, datasets_path, 8, 35)
                elif len(sentence.split(" ")) == 10:
                    model = prepare_model("model-9", first_language, second_language, datasets_path, 9, 40)
                elif len(sentence.split(" ")) == 11:
                    model = prepare_model("model-10", first_language, second_language, datasets_path, 10, 45)
                elif len(sentence.split(" ")) == 12:
                    model = prepare_model("model-11", first_language, second_language, datasets_path, 11, 50)

                sentence = re.sub(r"'", " ", sentence)

                output = model.predict(sentence).capitalize()
                output = re.sub(r" ([.!?])", re.search(
                    r"([.!?])", sentence).group(1), output)

                translate += output

        return translate

name = environ.get("TS_MODEL_NAME") or os.getenv("TS_MODEL_NAME")
first_language = environ.get("TS_FIRST_LANGUAGE") or os.getenv("TS_FIRST_LANGUAGE")
second_language = environ.get("TS_SECOND_LANGUAGE") or os.getenv("TS_SECOND_LANGUAGE")
datasets_path = environ.get("TS_DATASETS_PATH") or os.getenv("TS_DATASETS_PATH")
count_words = environ.get("TS_COUNT_WORDS") or os.getenv("TS_COUNT_WORDS") 
max_length = environ.get("TS_MAX_LENGTH") or os.getenv("TS_MAX_LENGTH") 
interation = environ.get("TS_INTERATION") or os.getenv("TS_INTERATION")
learning_rate = environ.get("TS_LEARNING_RATE") or os.getenv("TS_LEARNING_RATE")
hidden_size = environ.get("TS_HIDDEN_SIZE") or os.getenv("TS_HIDDEN_SIZE")
dropout_p = environ.get("TS_DROPOUT_P") or os.getenv("TS_DROPOUT_P")
teacher_forcing_ratio = environ.get("TS_TEACHER_FORCING_RATIO") or os.getenv("TS_TEACHER_FORCING_RATIO") 
print_every = environ.get("TS_PRINT_EVERY") or os.getenv("TS_PRINT_EVERY")

# Train a new model.

# if name is not None and count_words is not None and max_length is not None:
#     train_model(name, first_language, second_language, datasets_path, int(count_words), int(max_length), int(interation), float(learning_rate), int(hidden_size), float(dropout_p), float(teacher_forcing_ratio), int(print_every))

# Test the trained models using the command line.

# while(True):
#     print("Enter a sentence: ")
#     translated = translate(input(), first_language, second_language, datasets_path)
#     print(translated)

# Test the trained models using a REST API route.

@app.route("/translate", methods=["POST"])
def translate_text():
    return translate(request.data.decode("UTF-8"), first_language, second_language, datasets_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True, threaded=True)