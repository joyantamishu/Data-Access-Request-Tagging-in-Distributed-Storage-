import pandas as pd
import numpy as np
import math

process_map = dict()

process_map_track = dict()

global_sum = 0
df = pd.read_csv("madmax", names=["timestamp", "pid", "process",
                                  "lba", "size", "RW", "major", "minor", "md5"], low_memory=False
                 , delim_whitespace=True
                 )

print(df.shape[0])

unique_element_list = df['process'].unique()

file_csv = open("process_info_flowlet.csv", "w")
file_csv.write("process,access\n")


for element in unique_element_list:
    process_map[element] = 0
    process_map_track[element] = 0

real_count = 0
wrong_prediction_false_positive = 0
wrong_prediction_true_negative = 0
prev_ts = df['timestamp'].iloc[0]
prev_ps = df['process'].iloc[0]
prev_lba = df['lba'].iloc[0]
count = 0
threshold = 0.05

avg = 0.0

index_number = 1

string_op = ""

delimeter = '@'

def process_data(global_string, global_process_string):

    max_val = 0

    count = 0

    process = ""

    for element in unique_element_list:
        val = process_map[element] - process_map_track[element]
        count = count + val
        if val > max_val:
            process = element
            max_val = val

        process_map_track[element] = process_map[element]

    global_string_array = global_string.split(delimeter)
    global_process_string = global_process_string.split(delimeter)

    string_to_put = ""
    for x in range (0, len(global_string_array)):
        #if global_process_string[x] == process:
        string_to_put = string_to_put + global_string_array[x]+" "


    #print(string_to_put)

    file_csv.write(process+","+string_to_put+"\n")

    return count - max_val



global_string = ""
global_process_string = ""

for index, row in df.iterrows():
    diff = float(row['timestamp'] - prev_ts)/float(math.pow(10,9))

    process_change = 0

    guessed_change = 0

    process_map[row['process']] = process_map[row['process']] + 1

    string_to_insert = row['RW']

    if abs(row["lba"] - prev_lba) <= 8:
        string_to_insert = 'a'+string_to_insert

    else:
        string_to_insert = 'b' + string_to_insert

    #string_to_insert = string_to_insert + delimeter

    global_string = global_string + delimeter + string_to_insert

    global_process_string = global_process_string + delimeter + row['process']

    if row['process'] != prev_ps:
        process_change = 1
        real_count = real_count + 1

        avg = avg + diff

    if diff > threshold:
        guessed_change = 1
        count = count + 1

        global_sum = global_sum + process_data(global_string, global_process_string)

        global_string = ""
        global_process_string = ""



    if process_change == 1 and guessed_change==0:
        wrong_prediction_false_positive = wrong_prediction_false_positive +1

        #print(str(index_number) + row['process'] + " " + str(row['timestamp']))

    if process_change == 0 and guessed_change==1:
        wrong_prediction_true_negative = wrong_prediction_true_negative +1


    prev_ps = row['process']
    prev_ts = row['timestamp']
    prev_lba = row['lba']

    index_number = index_number + 1


print(str(threshold)+","+str(wrong_prediction_false_positive))
print(str(threshold)+","+str(wrong_prediction_true_negative))
print(real_count)
print(count)

print(avg/real_count)

print(global_sum)
