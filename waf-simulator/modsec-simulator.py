import time
import string
import numpy as np
import pandas as pd
import csv
import argparse
import pickle
import datetime
import random
from tqdm import tqdm
import requests

parser = argparse.ArgumentParser(description="Payload testing via remote ModSecurity")
parser.add_argument('--server', type=str, default="http://192.168.106.129/?id=", 
                    help="Base URL of Apache/ModSecurity test endpoint")
args = parser.parse_args()

def compute_score(payload):
    try:
        # Send the payload as a query parameter 'q'
        params = {"q": payload}
        response = requests.get(args.server, params=params, timeout=5)

        if response.status_code == 200:
            return 0
        elif response.status_code == 403:
            return 1
        else:
            print(f"Received unexpected HTTP code {response.status_code}; defaulting to 0.")
            return 0
    except Exception as e:
        print(f"Error computing score for payload: {e}")
        return 0.0

# --- Load SQLi Payloads from CSV ---
file_path = "gpt2.csv"
df = pd.read_csv(file_path, usecols=[1], names=["payloads"])
payloads = df["payloads"].tolist()
print(f"Loaded {len(payloads)} SQLi payloads.")

# file_path = "misp.txt"
# with open(file_path, "r", encoding="utf-8") as f:
#     payloads = [line.strip() for line in f.readlines()]  # strip to clean newlines
# print(f"Loaded {len(payloads)} SQLi payloads.")

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