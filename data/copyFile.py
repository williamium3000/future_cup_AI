import os
import shutil

fo = open('training.csv', 'r')
label_list = []
img_path = []
for line in fo:
    line = line.replace('\n', '')
    img_path.append(line.split(',')[0]+'.jpg')
    label_list.append(line.split(',')[1])
fo.close()

fo2 = open('annotation.csv', 'r')
label_list2 = []
img_path2 = []
for line in fo2:
    line = line.replace('\n', '')
    img_path2.append(line.split(',')[0]+'.jpg')
    label_list2.append(line.split(',')[1])
fo2.close()


fo1 = open('test.csv', 'r')
label_list1 = []
img_path1 = []
for line in fo1:
    line = line.replace('\n','')
    img_path1.append(line + '.jpg')
    label_list1.append('21')
fo1.close()

for i in range(len(img_path2)):
    if img_path2[i] in img_path1:
        for j in range(1,len(img_path1)):
            if img_path2[i] == img_path1[j]:
                label_list1[j] = label_list2[i]




for i in range(20):      #创建文件夹
    os.mkdir('training' + str(i))

for i in range(20):      #创建文件夹
    os.mkdir('test' + str(i))

for i in range(1, len(label_list)):      #train
    shutil.copy('data/' + img_path[i], './training' + label_list[i] + '/' + img_path[i])

for i in range(1,len(label_list1)):
    shutil.copy('data/'+img_path1[i],'./test' + label_list1[i] + '/' + img_path1[i])