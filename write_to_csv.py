import csv
import os
import re

def parse_data(output_text):
    data_records = {}
    lines = output_text.strip().split('\n')
    
    current_record = {}
    
    for line in lines:
        # Use regex to find all key-value pairs in the line
        pairs = re.findall(r'(\S+)\s+(-?\d\.?\d*e?[+\-]?\d*)', line)
        
        for key, value in pairs:
            # Clean up the key and convert the value to a float or int
            key = key.replace('_', ' ').replace('|', '').strip()
            try:
                # Handle scientific notation and regular numbers
                value = float(value)
                if value == int(value):
                    value = int(value)
            except ValueError:
                # If conversion fails, keep as string
                pass
            
            # Use a unique identifier (i and ii) for each record
            if key == 'i' or key == 'ii':
                current_record[key] = value
            else:
                current_record[key] = value

        # When a new record starts, save the old one
        if 'i' in current_record and 'ii' in current_record:
            record_id = (current_record['i'], current_record['ii'])
            if record_id not in data_records:
                data_records[record_id] = {}
            data_records[record_id].update(current_record)
            current_record = {}
    
    return list(data_records.values())


def write_to_csv(data, filename):
    if not data:
        return
        
    # Get the union of all keys to create the header
    all_keys = set().union(*(d.keys() for d in data))
    fieldnames = sorted(list(all_keys))

    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write the header row
        writer.writeheader()
        
        # Write each data record
        for row in data:
            writer.writerow(row)

# Example usage with your data
with open('output.txt', 'r') as file:
    output_text = file.read()

# 1. Parse the text
parsed_data = parse_data(output_text)

# 2. Write the parsed data to a CSV file
write_to_csv(parsed_data, 'output_data.csv')

print("CSV file 'output_data.csv' has been created.")

# read csv file to find max value of reward_sum
import pandas as pd
df = pd.read_csv('output_data.csv')
max_reward = df['reward sum'].max()
print(f'Max reward sum: {max_reward}')