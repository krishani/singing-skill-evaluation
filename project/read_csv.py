import csv
times=[]
durations=[]
notes=[]

with open('ayanna_kiyanna.notes.csv','rb')as f:
	reader=csv.reader(f)
	for row in reader:
		
		times.append(round(float(row[0]),3))
		durations.append(round(float(row[1]),3))
		notes.append(row[2])
	for i in range(len(times)):
		print times[i]
		print "time"
	
