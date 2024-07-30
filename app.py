import os
import re

from config import Config
from flask import Flask, request
from seq2seq import Seq2Seq
from utils.prepare_data import PrepareData

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
    data = PrepareData(first_language, second_language,
                       datasets_path, count_words)
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
    data = PrepareData(first_language, second_language,
                       datasets_path, count_words)
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
                    text_split[i - 1].lower(),  text_split[i]).strip()

                if len(sentence.split(" ")) == 2:
                    model = prepare_model(
                        "model-1", first_language, second_language, datasets_path, 1, 5)
                elif len(sentence.split(" ")) == 3:
                    model = prepare_model(
                        "model-2", first_language, second_language, datasets_path, 2, 10)
                elif len(sentence.split(" ")) == 4:
                    model = prepare_model(
                        "model-3", first_language, second_language, datasets_path, 3, 10)
                elif len(sentence.split(" ")) == 5:
                    model = prepare_model(
                        "model-4", first_language, second_language, datasets_path, 4, 15)
                elif len(sentence.split(" ")) == 6:
                    model = prepare_model(
                        "model-5", first_language, second_language, datasets_path, 5, 20)
                elif len(sentence.split(" ")) == 7:
                    model = prepare_model(
                        "model-6", first_language, second_language, datasets_path, 6, 25)
                elif len(sentence.split(" ")) == 8:
                    model = prepare_model(
                        "model-7", first_language, second_language, datasets_path, 7, 30)
                elif len(sentence.split(" ")) == 9:
                    model = prepare_model(
                        "model-8", first_language, second_language, datasets_path, 8, 35)
                elif len(sentence.split(" ")) == 10:
                    model = prepare_model(
                        "model-9", first_language, second_language, datasets_path, 9, 40)
                elif len(sentence.split(" ")) == 11:
                    model = prepare_model(
                        "model-10", first_language, second_language, datasets_path, 10, 40)
                elif len(sentence.split(" ")) == 12:
                    model = prepare_model(
                        "model-11", first_language, second_language, datasets_path, 11, 45)
                elif len(sentence.split(" ")) == 13:
                    model = prepare_model(
                        "model-12", first_language, second_language, datasets_path, 12, 50)
                elif len(sentence.split(" ")) == 14:
                    model = prepare_model(
                        "model-13", first_language, second_language, datasets_path, 13, 55)
                elif len(sentence.split(" ")) == 15:
                    model = prepare_model(
                        "model-14", first_language, second_language, datasets_path, 14, 60)
                elif len(sentence.split(" ")) == 16:
                    model = prepare_model(
                        "model-16", first_language, second_language, datasets_path, 15, 65)
                else:
                    model = None

                sentence = re.sub(r"'", " ", sentence.replace(",", ""))

                if model is not None:
                    output = model.predict(sentence).capitalize()
                    output = re.sub(r" ([.!?])", re.search(
                        r"([.!?])", sentence).group(1), output)

                    translate += output

        return translate


# Parameters of the model.

config = Config()

name = config.getenv("TS_MODEL_NAME")
first_language = config.getenv("TS_FIRST_LANGUAGE")
second_language = config.getenv("TS_SECOND_LANGUAGE")
datasets_path = config.getenv("TS_DATASETS_PATH")
count_words = config.getenv("TS_COUNT_WORDS")
max_length = config.getenv("TS_MAX_LENGTH")
interation = config.getenv("TS_INTERATION")
learning_rate = config.getenv("TS_LEARNING_RATE")
hidden_size = config.getenv("TS_HIDDEN_SIZE")
dropout_p = config.getenv("TS_DROPOUT_P")
teacher_forcing_ratio = config.getenv("TS_TEACHER_FORCING_RATIO")
print_every = config.getenv("TS_PRINT_EVERY")

# Train a new model.

# if name is not None and count_words is not None and max_length is not None:
#     train_model(name, first_language, second_language, datasets_path, int(count_words), int(max_length), int(
#         interation), float(learning_rate), int(hidden_size), float(dropout_p), float(teacher_forcing_ratio), int(print_every))

# Test the trained models using the command line.

# while (True):
#     print("Enter a sentence: ")
#     translated = translate(input(), first_language,
#                            second_language, datasets_path)
#     print(translated)

# Test the trained models using a REST API route.


@app.route("/translate", methods=["POST"])
def translate_text():
    return translate(request.data.decode("UTF-8"), first_language, second_language, datasets_path)


if __name__ == "__main__":
    app.run(host=os.getenv("HOST"), port=os.getenv(
        "PORT"), debug=True, threaded=True)
