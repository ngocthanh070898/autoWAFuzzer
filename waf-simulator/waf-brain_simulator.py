import time
import string
import numpy as np
import pandas as pd
import csv
from keras.models import load_model
import argparse
import pickle
import random
from tqdm import tqdm
import requests

X_FEATURES = 5
BATCH_SIZE = 100000
EPOCHS = 200
TRAIN_PERC = 0.7
DEV_PERC = 0.25
TEST_PERC = 0.05
VOCABULARY = string.printable

# parser = argparse.ArgumentParser(description="")
# parser.add_argument('--lm_name')
# parser.add_argument('--ref_lm_name')
# parser.add_argument('--total_nums')
# args = parser.parse_args()

# --- WAF-Brain---
def row_parse(row):
    def char_parse(char):
        return VOCABULARY.index(char)
    return [char_parse(char) for char in row]

def reduce_dimension(row, x_features):
    def create_new_row(index, char):
        past_pos = x_features - 1
        new_row = [None for i in range(x_features + 1)]

        def fill_past():
            if index > past_pos:
                for i in range(past_pos * -1, 0, 1):
                    pos = index - i
                    if pos:
                        new_row[past_pos + i] = row[index + i]
            else:
                for i in range(x_features, 0, -1):
                    if (index - i) >= 0:
                        x = (index - i) + 1
                        new_row[past_pos - x] = row[index - x]

        def fill_future():
            if (index + 1) < len(row):
                new_row[-1] = row[index + 1]

        fill_past()
        new_row[past_pos] = char
        fill_future()
        return new_row
    return [create_new_row(index, char) for index, char in enumerate(row)]

def split_row(x, y, row):
    for t in row:
        x_zeros = np.zeros((X_FEATURES, 101))
        y_zeros = np.zeros((101))
        get_index = lambda index: 100 if index is None else index
        for i in range(X_FEATURES):
            x_zeros[i][get_index(t[i])] = 1
        x.append(x_zeros)
        y_zeros[get_index(t[-1])] = 1
        y.append(y_zeros)

def to_ascii(row):
    for i, elem in enumerate(row):
        if elem == 1:
            return VOCABULARY[i]

def transform_predict(y_predict):
    index = np.argmax(y_predict)
    if index == 100:
        return None
    return VOCABULARY[index]

def build_text(predict_char, predict_texts):
    if predict_char is None:
        predict_texts.append([])
    else:
        predict_texts[-1].append(predict_char)

# --- Load the Keras Model for Scoring ---
keras_model = load_model('waf-brain.h5')
keras_model.summary()

def compute_score(payload):
    try:
        before_time = time.time()
        # Process the payload: parse characters, reduce dimension, then split into training format.
        elems = [reduce_dimension(row_parse(payload), X_FEATURES)]
        x_demo = []
        y_demo = []
        for elem in elems:
            split_row(x_demo, y_demo, elem)
        # Evaluate using the Keras model (assumes model.evaluate returns [loss, accuracy], and we take accuracy)
        score = keras_model.evaluate(np.array(x_demo), np.array(y_demo), verbose=0)[1]
        diff_time = time.time() - before_time
        return round(score, 2)
    except Exception as e:
        print(e)
        return None

# --- Load SQLi Payloads from TXT ---
# file_path = "new-sqli.txt"
# df = pd.read_csv(file_path, names=["payloads"], nrows=1000, on_bad_lines="skip")
# payloads = df["payloads"].tolist()
# print(f"Loaded {len(payloads)} SQLi payloads.")

# --- Load SQLi Payloads from CSV ---
file_path = "gpt2.csv"
df = pd.read_csv(file_path, usecols=[1], names=["payloads"])
payloads = df["payloads"].tolist()
print(f"Loaded {len(payloads)} SQLi payloads.")

# --- Build Metadata Dictionary (including computed score) ---
csv_filename = "score-gpt2-only.csv"
with open(csv_filename, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    for i, payload in enumerate(tqdm(payloads, desc="Processing payloads")):
        label = compute_score(payload)
        status = "bypassed" if label == 0 else "blocked"
        writer.writerow([payload, label, status])

# --- Load the CSV file and compute metrics ---
score = pd.read_csv(csv_filename, names=['payloads', 'score', 'status'])

mean_score = score['score'].mean()
num_bypassed = score[score['status'] == 'bypassed'].shape[0]
percent_bypassed = round((num_bypassed / score.shape[0]) * 100, 2)

# Create a DataFrame to combine the results
results = pd.DataFrame({
    "Metrics": ["Mean Score", 
               "Number of Bypassed Payloads", "Total Number of Payloads", "Percentage of Bypassed Payloads"],
    "Model": [mean_score, num_bypassed, score.shape[0], f"{percent_bypassed}%"],
})

print(results.head())