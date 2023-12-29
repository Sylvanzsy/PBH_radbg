import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
import mpl_toolkits.mplot3d

formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1,1))

#plt.style.use('test2')

# read data to file as rows of strings separated by ' '
# s: file name, k: num. of head rows to skip
def restr(s, k = 0):
	out = []
	j = 0
	with open(s, 'r') as f:
		for line in f:
			lst = line.split()
			if j<k:
				a=1#print (lst[0])
			else:
				out.append(lst)
			j = j+1
	return out

# read data to file as rows of floats...
def refloat(s, k = 0):
	out = []
	j = 0
	with open(s, 'r') as f:
		for line in f:
			lst = line.split()
			if j<k:
				a=1#print (lst[0])
			else:
				out.append(np.array([float(x) for x in lst]))
			j = j+1
	return out

# read data to a list of arrays, each array is one column
# s: file name, n: num. of columns, k: num. of head rows to skip, t: reverse or not
def retxt(s, n, k = 0, t = 0, column_index = -1):
	out = []
	for i in range(n):
		out.append([])
	j = 0
	with open(s, 'r') as f:
		for line in f:
			lst = line.split()
			if j<k:
				a=1#print (lst[0])
			else:
				for i in range(n):
					out[i].append(float(lst[i]))
			j = j+1
	if t!=0:
		for i in range(n):
			out[i].reverse()
	out[i] = np.array(out[i])
	if column_index !=-1:
		return  out[column_index]
	return out
# similar to retxt but skip the rows in which the column defined by ind 
# is not in ascending order, pre0 sets the minimum of this column
def retxt_nor(s, n, k = 1, t = 1, ind = 0, pre0 = 0):
	out = []
	for i in range(n):
		out.append([])
	j = 0
	pre = pre0
	with open(s, 'r') as f:
		for line in f:
			lst = line.split()
			if j<k:
				a=1#print (lst[0])
			else:
				if float(lst[ind]) >= pre:
					for i in range(n):
						out[i].append(float(lst[i]))
					pre = float(lst[ind])
			j = j+1
	if t!=0:
		for i in range(n):
			out[i].reverse()
	out[i] = np.array(out[i])
	return out

# write a list of arrays to file as columns
# s:file name, l: input list, ls: head row, k: num. of rows to skip, t: head or not
def totxt(s, l, ls = 0, t = 0, k = 0):
	j = 0
	with open(s, 'w') as f:
		if t!=0:
			for r in range(len(ls)):
				f.write(ls[r])
				f.write(' ')
			f.write('\n')
		for i in range(len(l[0])):
			if j<k:
				print (l[0][i])
			else:
				for s in range(len(l)):
					f.write(str(l[s][i]))
					f.write(' ')
				f.write('\n')
			j = j+1

# write a list of arrays to file as rows
def torow(s, l, k=0):
	j = 0
	with open(s, 'w') as f:
		for row in l:
			if j<k:
				print(row)
			else:
				for d in row:
					f.write(str(d))
					f.write(' ')
				f.write('\n')
			j = j+1
