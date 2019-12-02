# -*- coding: UTF-8 -*- 

"""
Spyder Editor:nihao

This is a temporary script file.
"""




import json

with open("mgn.json", 'r') as f1:
    b = json.load(f1)
with open("BDB.json", 'r') as f2:
    a = json.load(f2)

print(len(a))
print(len(b))
c={}
k=[]

for key in a.keys():
    temp = []
    flag = 1
    i = 0
    while len(temp) != len(a[key]):
        if a[key][i] not in temp:
            temp.append(a[key][i])
        if a[key][i+1] not in temp and len(temp) < len(a[key]):
            temp.append(a[key][i+1])
        if len(temp) < len(a[key]) and a[key][i+2] not in temp:
            temp.append(a[key][i+2])
        if len(temp) < len(a[key]) and a[key][i+3] not in temp:
            temp.append(a[key][i+3])
        if len(temp) < len(a[key]) and a[key][i+4] not in temp:
            temp.append(a[key][i+4])
        if b[key][int(i/5)] not in temp and len(temp) < len(a[key]):
            temp.append(b[key][int(i/5)])
        i = i+5
    c[key] = temp
    k.append(i)
    

print(len(c))
print(k)


with open("final_battle.json", 'w', encoding='utf-8') as f:
    json.dump(c,f)
    print("complete")













