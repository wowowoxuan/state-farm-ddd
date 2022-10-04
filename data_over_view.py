import csv
count = {}
overall = {}
total_num = 0

csvFile = open('/media/weiheng/Elements/Data/state-farm-distracted-driver-detection/driver_imgs_list.csv')
reader = csv.reader(csvFile)
for item in reader:
    
    if reader.line_num == 1:
        continue
    total_num += 1
    if not item[0] in overall:
        overall[item[0]] = 1
    else:
        overall[item[0]] += 1
    if item[0] not in count.keys():
        count[item[0]] = {}
        count[item[0]][item[1]] = 1
    else:
        if item[1] in count[item[0]].keys():
            count[item[0]][item[1]] += 1
        else:
            count[item[0]][item[1]] = 1
# print(count)
print(len(overall))
print(total_num)




