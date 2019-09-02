# -*- coding:utf-8 _*-
""" 
@file: statistic.py 
@time: 2019/7/13
"""


def seg_layer(boundary):
    layer_max = []
    layer = []
    for b in boundary:
        left, right = int(b[0]), int(b[1])
        flag = True
        for i in range(len(layer_max)):
            if left >= layer_max[i]:
                layer_max[i] = right
                layer[i].append((left, right))
                flag = False
                break
        if flag:
            layer_max.append(right)
            layer.append([(left, right)])
    return layer


with open("./dataset/ACE05/train/train.data") as f:
    all_data = f.read().strip().split("\n\n")
    for line in all_data:
        infos = line.strip().split("\n")
        if len(infos) != 3:
            continue

        boundary = [item.split(" ")[0].split(",")[:2] for item in infos[2].split("|")]
        boundary = sorted(boundary, key=lambda x: (int(x[1]), -int(x[0])))
        layer = seg_layer(boundary)
        if len(layer) == 3:
            print(infos[0])
