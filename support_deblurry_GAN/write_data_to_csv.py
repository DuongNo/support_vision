import csv
import time
with open('names.csv', 'w+') as csvfile:
    fieldnames = ['Epochs', 'Generator_Loss', 'Discriminator']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    csvfile.close()
# while(1):
a = 999
with open('names.csv', 'a') as csvfile:
    fieldnames = ['Epochs', 'Generator_Loss', 'Discriminator']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    #writer.writeheader()
    #writer.close()

    writer.writerow({'Epochs': a, 'Generator_Loss': a, 'Discriminator':a})
    writer.writerow({'Epochs': '2', 'Generator_Loss': '0.33', 'Discriminator':'0.63'})
    writer.writerow({'Epochs': '3', 'Generator_Loss': '0.16', 'Discriminator':'.074'})
    csvfile.close
    time.sleep(4)

# import collections
# columns = collections.defaultdict(list)ss

# values = []
# with open("/home/duong/Documents/researching/GAN/common/dl/names.csv", "r") as f:
#     reader = csv.reader(f)
#     i = next(reader) #reader.next()
# with open("/home/duong/Documents/researching/GAN/common/dl/names.csv", "r") as csv_file:
#     csv_reader = csv.DictReader(csv_file, delimiter=',')
#     for lines in csv_reader:
#         values.append(int(lines[i[0]]))
# print(values)

# print(columns[i[0]])
# print(columns[i[1]])