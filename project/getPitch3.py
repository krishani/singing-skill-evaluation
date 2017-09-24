from subprocess import Popen, PIPE, call
from scipy import linspace, polyval, polyfit, sqrt, stats, randn,signal
from collections import Counter
from sklearn import datasets, linear_model
import numpy as np
import matplotlib.pyplot as plt
import math

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

text = toPitch('ayanna_kiyanna.wav')
pitchDic,timelist,pitchlist = read_praat_out(text)

pitches=np.asarray(pitchlist, dtype=float)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#preprocessing 
#Create an order 3 lowpass butterworth filter:


b, a = signal.butter(3, 0.05)
#Apply the filter to xn. Use lfilter_zi to choose the initial condition of the filter:
xn=pitches

zi = signal.lfilter_zi(b, a)
z, _ = signal.lfilter(b, a, xn, zi=zi*xn[0])

#Apply the filter again, to have a result filtered at an order the same as filtfilt:


z2, _ = signal.lfilter(b, a, z, zi=zi*z[0])

#Use filtfilt to apply the filter:


y = signal.filtfilt(b, a, xn)

pitches=y
print pitches
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%





#remove silent sections
index = np.argwhere(pitches==0.0)
#print index
pitches_with_no_zero = np.delete(pitches, index)

print 'pitches with no 0000000000000000000000000000000000'

#print pitches_with_no_zero

cent_vals=[]
offset_vals=[]
frame_vals=[]

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

################################################################
convert_to_cents(pitches_with_no_zero)

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
######################################################################

get_offset(cent_vals)

cents_original=np.asarray(offset_vals, dtype=float)

time_original=np.arange(0,len(cents_original),1)

#print len(cents_original)//5

def plotgraph(cent_ar,time_ar,xlabel,ylabel,title):

	plt.plot(time_ar,cent_ar,'r.')
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.tight_layout() 
	plt.title(title)
	
	plt.legend()

	plt.show()

plotgraph(cents_original,time_original,'time msec','offset','OFFSETS')

def break_into_frames(frames):
	frame_size=10
	for i in range (frames):
		start_in=i
		finish_in=i+frame_size
		if(finish_in>frames):
			print 'finish index is > length'
			#check the last part
		else:
			temp=np.array(cents_original[start_in:finish_in])
			med= np.median(temp)
			frame_vals.append(med)
	return frame_vals

def break_into_frames2(pitch_ar,size):
	#t=np.arange(0,len(pitch_ar),size)
	#print len(t)
	print pitch_ar[0:10]
	print pitch_ar[10:20]
	j=1
	
	for i in range(0,len(pitch_ar),size):
		print i
		temp=np.asarray(pitch_ar[i:j*size])
		med=np.median(temp)
		frame_vals.append(med)
		#print temp
		#print med
		j+=1
	print frame_vals
	return frame_vals
		
		
		
	
		
#break_into_frames(len(cents_original)//5)
#print(frame_vals)
#print(len(frame_vals))	

break_into_frames2(cents_original,10)

###################make a frame array

#####################


frame_vals.append(100.0)
#####################added this one point further to represent 100
frames=np.arange(0,len(frame_vals),1)


plotgraph(frame_vals,frames,'frames','offset','OFFSETS')
########################
##############################

