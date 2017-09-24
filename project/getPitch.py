from subprocess import Popen, PIPE, call
from scipy import linspace, polyval, polyfit, sqrt, stats, randn
from collections import Counter
from sklearn import datasets, linear_model
import numpy as np
import matplotlib.pyplot as plt
import math

cent_vals=[]
cent_vals_no_zero=[]
offset_vals=[]

def toPitch(filename):
    p = Popen(['/usr/bin/praat','--run','pitch.praat',filename],stdout = PIPE,stderr = PIPE,stdin = PIPE)
    stdout, stderr = p.communicate()
    return stdout.decode().strip()
def read_praat_out(text):
    if not text:
        return None
    lines = text.splitlines()
    head = lines.pop(0)
    head = head.split("\t")[1:]
    output = {}
    outputlist = []
    valueslist = []
    for l in lines:
        if '\t' in l:
            line = l.split("\t")
            time = line.pop(0)
            values = {}
            
            for j in range(len(line)):
                v = line[j]
                if v != '--undefined--':
                    try:
                        v = float(v)
                    except ValueError:
                        print(text)
                        print(head)
                else:
                    v = 0
                values[head[j]] = v
                valueslist.append(v)
            if values:
                output[float(time)] = values
                outputlist.append(time)
    return output,outputlist,valueslist

text = toPitch('organ.wav')
pitchDic,timelist,pitchlist = read_praat_out(text)
#print pitchlist

def convert_to_cents(pitchlist):
#this function will take the original f0 values and converts it to cents
	for pitch in pitchlist:
		if(pitch==0.0):
			x=pitch
		else:
			
			x=int(round(1200*math.log(pitch/(440*math.pow(2,3/12-5)),2)))
		
		cent_vals.append(x)
	print cent_vals
	return cent_vals
def remove_silent(cent_vals):

	for pitch in cent_vals_no_zero:
		if(pitch!=0.0):
			
		
		
			cent_vals_no_zero.append(pitch)
	print cent_vals_no_zero
	return cent_vals_no_zero
	

def plotgraph(cent_ar,time_ar,xlabel,ylabel,title):

# Plot the data
	plt.plot(time_ar,cent_ar,'r.')
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.tight_layout() 
	plt.title(title)
	#fig = plt.figure(figsize=(20, 2))
	#ax = fig.add_subplot(111)
	#ax.plot(x, y)

# Add a legend
	plt.legend()

# Show the plot
	plt.show()
def get_offset(cent_vals):
	for pitch in cent_vals:
		x=pitch%100
		if(x>50):
			offset=100-x
		else:
			offset=x
			
		
		offset_vals.append(offset)
	print offset_vals
	return offset_vals
def linear_regression(x,y):
	(ar,br)=polyfit(x,y,1)
	xr=polyval([ar,br],y)
	err=sqrt(sum((xr-y)**2)/len(x))
	plt.title('Linear Regression')
	
	plt.plot(x,y,'k.')
	plt.plot(x,xr,'r.-')
	plt.legend(['original', 'regression'])
	plt.show()
	return err
def linear_regression2(x,y,testing_set):
	size_of_training_set=int(round(len(x)/testing_set)*(1))
	print size_of_training_set
	
	x=x.reshape(len(x),1)
	y=y.reshape(len(y),1)
 
	
 


	x_train = x[:-size_of_training_set]
	x_test = x[-size_of_training_set:]

	y_train = y[:-size_of_training_set]
	y_test = y[-size_of_training_set:]
	#print len(x_train)
	#print len(x_test)

	regr =linear_model.LinearRegression()
	regr.fit(x_test,y_test)
	pred=regr.predict(x_test)
	coef=regr.coef_
	print coef
	plt.title('offsets with linear regression')
	
	plt.plot(x,y,'k.')
	plt.plot(x_test,pred,color='blue',linewidth=3)
	plt.show()
	
	
	return regr.coef
		
	
convert_to_cents(pitchlist)
#remove_silent(cent_vals)
#plotgraph(cent_vals,timelist,'time msec','frequency cents','FO TRAJECTORY')
#get_offset(cent_vals)
get_offset(cent_vals)
#plotgraph(offset_vals,timelist,'time msec','offset','OFFSETS')



new_time=np.arange(0,len(cent_vals_no_zero),1)
x=np.asarray(timelist, dtype=float)
y=np.asarray(offset_vals, dtype=float)
linear_regression2(x,y,1)
#print timelist


