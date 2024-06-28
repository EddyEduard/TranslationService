from utils.prepare_data import PrepareData
from seq2seq import Seq2Seq
from flask import Flask, request

import re

app = Flask(__name__)

'''

Train the model.

name - name of trained model
first_language - first language name
second_language - second language name
datasets_path - datasets path
count_words - number of the words
max_length - maximum sentence length

'''

def train_model(name, first_language, second_language, datasets_path, count_words, max_length):
    data = PrepareData(first_language, second_language, datasets_path, count_words)
    source_sentence, target_sentence, sentence_pairs = data.process()

    seq2seq = Seq2Seq(interation=1000,
                      learning_rate=0.01,
                      hidden_size=256,
                      max_length=max_length,
                      dropout_p=0.1,
                      teacher_forcing_ratio=0.5,
                      print_every=100
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

                sentence = re.sub(r"'", " ", sentence)

                output = model.predict(sentence).capitalize()
                output = re.sub(r" ([.!?])", re.search(
                    r"([.!?])", sentence).group(1), output)

                translate += output

        return translate

# Train a  new model.

# train_model("model-7", "English", "Romanian", "data/en-ro.txt", 7, 30)

# Test the trained models using the command line.

while(True):
    print("Enter a sentence: ")
    translated = translate(input(), "English", "Romanian", "data/en-ro.txt")
    print(translated)

# Test the trained models using a REST API route.

# @app.route("/translate", methods=["POST"])
# def translate_text():
#     return translate(request.data.decode("UTF-8"), "English", "Romanian", "data/en-ro.txt")

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=8080, debug=True, threaded=True)
