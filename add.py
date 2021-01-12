import numpy as np
'a binary search tree'
l = np.arange(0, 30, 3)
m = np.median(l)
tree1 = []
tree2 = []
tree3 = []
tree4 = []
tree5 = []
tree6 = []
for i in l:
    if i > m :
        tree1.append(i)
    else :
        tree2.append(i)
for j in tree1:
    if j > np.median(tree1) :
        tree3.append(j)
    else:
        tree4.append(j)
for k in tree3:
    if k > np.median(tree3):
        tree5.append(k)
    else:
        tree6.append(k)
print(tree1)
print(tree3)
print(tree5)
print(tree4)
print(tree6)


