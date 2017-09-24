w, h = 8, 5;
Matrix = [[0 for x in range(w)] for y in range(h)] 

for i in range(h):
	k=1
	for j in range(w):
		
		Matrix[i][j]=k
		k+=1
print Matrix
