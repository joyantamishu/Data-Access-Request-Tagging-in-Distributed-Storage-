import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("flowlet_csv.csv")


list_threshold_x = list()

list_error_percentage_y = list()
list_whole_error_percentage_y = list()

def normalize_array(list_to_normlize):

    max_v = max(list_to_normlize)

    min_v = min(list_to_normlize)

    for x in range(0, len(list_to_normlize)):
        list_to_normlize[x] = float(list_to_normlize[x] - min_v)/float(max_v-min_v)

    return list_to_normlize

count = 0
for index, row in df.iterrows():
    list_threshold_x.append(row['threshold'] * 1000)
    list_error_percentage_y.append(row['error_percentage'])
    list_whole_error_percentage_y.append(row['whole_accuracy'])
    count = count + 1


list_whole_error_percentage_y = normalize_array(list_whole_error_percentage_y)
#x = [1, 2, 3]
# corresponding y axis values
#y = [2, 4, 1]

# plotting the points


plt.subplot(2, 1, 1)
plt.plot(list_threshold_x, list_error_percentage_y)

# naming the x axis
plt.xlabel('Threshold (in Millisecond)')
# naming the y axis
plt.ylabel('Error Rate %')

# giving a title to my graph
plt.title('Threshold vs Error Rate using Flowlet')


plt.subplot(2, 1, 2)
plt.plot(list_threshold_x, list_whole_error_percentage_y)

# naming the x axis
plt.xlabel('Threshold (in Millisecond)')
# naming the y axis
plt.ylabel('Accuracy')

# giving a title to my graph
plt.title('Threshold vs Accuracy Flowlet')
plt.tight_layout()


# function to show the plot
plt.show()
