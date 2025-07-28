import os
import json
import csv
import glob
import argparse
from collections import Counter


parser = argparse.ArgumentParser(description="Generate CSV for metrics")
parser.add_argument('--folder_path', type=str, required=True, help="Path to the folder containing JSON files")
parser.add_argument('--output_path', type=str, required=True, help="Path to the output CSV file")
args = parser.parse_args()

folder_path = args.folder_path
output_path = args.output_path

json_files = sorted(glob.glob(os.path.join(folder_path, '*.json')))
content_files = sorted(glob.glob(os.path.join(folder_path, '*.jsonl')))
parsed_decoded_text_counter = Counter()

csv_headers = ['task_name', 'metric', 'value']

with open(output_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(csv_headers)
    
    for file_path in json_files:
        file_name = os.path.basename(file_path)
        task_name = file_name.split('_')[0]
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        if 'acc' in data:
            metric = 'acc'
            value = data['acc']
        else:
            metric = next((m for m in ['em', 'rl', 'f1',] if m in data), None)
            value = data.get(metric)
        
        if metric and value is not None:
            writer.writerow([task_name, metric, round(value, 2)])

 
for file_path in content_files:
    with open(file_path, 'r') as f:
        for line in f:
            json_data = json.loads(line.strip())
            parsed_decoded_text = json_data.get('parsed_decoded_text')
            if parsed_decoded_text:
                parsed_decoded_text_counter[parsed_decoded_text] += 1

with open(output_path, 'a', newline='') as csvfile:  
    writer = csv.writer(csvfile)
    writer.writerow(['parsed_decoded_text', 'count'])  
    for parsed_text, count in parsed_decoded_text_counter.items():
        writer.writerow([parsed_text, count])

