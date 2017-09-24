
import numpy as np
from scipy.signal import butter, lfilter, freqz
import matplotlib.pyplot as plt
from subprocess import Popen, PIPE, call
from scipy import linspace, polyval, polyfit, sqrt, stats, randn
from collections import Counter
from sklearn import datasets, linear_model
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy as sp
from scipy.signal import kaiserord, lfilter, firwin, freqz
import hz_to_cents
import combfilter
import random
from scipy.interpolate import Rbf, InterpolatedUnivariateSpline
import intpolate
import time

cent_vals=[]

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
directory="sounds/"
sound_name='ayanna_kiyanna'
file_name=directory+sound_name+'.wav'
text = toPitch(file_name)
pitchDic,timelist,pitchlist = read_praat_out(text)

y=np.asarray(pitchlist,dtype=float)
x=np.asarray(timelist,dtype=float)

sample_rate = 44.100
nyq_rate = sample_rate / 2.0
width = 20.0/nyq_rate
ripple_db = 100.0
N, beta = kaiserord(ripple_db, width)
cutoff_hz = 5.0
taps = firwin(N, cutoff_hz/nyq_rate, window=('kaiser', beta))
filtered_x = lfilter(taps, 1.0, y)

#print filtered_x

delay = 0.5 * (N-1) / sample_rate
plt.plot(x, y,'b',linewidth=2)
#plt.plot(x-delay, filtered_x, 'r-',linewidth=0.5)
plt.plot(x[N-1:]-delay, filtered_x[N-1:], 'r-', linewidth=0.5)
plt.legend(['noisy', 'filtered'])
plt.show()

#t=np.arange(0,len(filtered_x[N-1:]))
#plt.plot(t,filtered_x[N-1:])
#plt.show()

fil=filtered_x[N-1:]
for i in range(len(fil)):
	if(fil[i]<20):
		fil[i]=0
		


pitches=np.asarray(fil, dtype=float)
#remove silent sections
index = np.argwhere(pitches==0.0)
#print index
pitches_with_no_zero = np.delete(pitches, index)
t=np.arange(0,len(pitches_with_no_zero))
'''
plt.plot(t,pitches_with_no_zero)
plt.xlabel('time')
plt.ylabel('frequency hz')
plt.show()
'''
hz_to_cents.convert_to_cents(pitches_with_no_zero,cent_vals)
cents=np.asarray(cent_vals, dtype=float)
'''
plt.plot(t,cents)
plt.xlabel('time')
plt.ylabel('frequency cents')
plt.show()
'''
f,axarr=plt.subplots(2,sharex=True)
axarr[0].plot(t,pitches_with_no_zero)
axarr[0].set_ylabel('hz')
axarr[1].plot(t,cents)
axarr[1].set_ylabel('cents')
plt.show()



peaks=[]
'''
def ss_in_a_frame(frame,F_val):
	for F in range(100):
		sigma=0
		for x in range(len(frame)):
			sigma+=combfilter.gaussianComb(frame[x],F)
	
		F_val.append(sigma)
	return F_val
'''

start_index=0
num_of_points=len(cents)
frame_size=200
incr=0
ss_frame=[]

	
start_time=time.time()	
count_ar=[]
for i in range(100):
	count_ar.append(0)

num_of_frames=0
#########################################################################################

for i in range(0,num_of_points-frame_size,50):
#for i in range(0,500,100):
	F_val=[]
	frame=cents[i:frame_size+i]
	#print 'frameeeeeeeeeee start ------->'+str(i)

	for F in range(100):
		#print 'FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF----->'+str(F)
		sigma=0
		for x in range(len(frame)):
			y=combfilter.gaussianComb(frame[x],F)
			sigma+=y
			#print 'x->'+ str(frame[x])+'F->'+str(F)+' gaussian value x,f->'+str(y)
	
		F_val.append(sigma)
		#print 'gaussian val for frame for F->'+str(sigma)
	
	#print F_val
	#plt.plot(F_val,range(100))
	#plt.show()
	########################################add the values FRAME1 ->F=0,F=1,F=2.............................
	for k in range(len(F_val)):
		count_ar[k]+=F_val[k]
	##########################################################################################33
	npFval=np.asarray(F_val,dtype=float)
	#peak=np.argmax(npFval)
	print 'frame start'+str(i)+' end->'+str(frame_size+i)
	#peaks.append(peak)
	num_of_frames+=1



########################################################################
npcount=np.asarray(count_ar,dtype=float)
gf=npcount/num_of_frames
'''
p=intpolate.fit(range(100),npcount)
xnew=np.arange(-10,110,1)
ynew=p(xnew)

#plt.plot(x,y,'b-')
#plt.show()
plt.plot(ynew,xnew,'b-')
plt.show()
'''
print "My program took", time.time() - start_time, "to run"
plt.plot(gf,range(100))
plt.xlabel('g(F)')
plt.ylabel('cent')
#plt.xticks(np.arange(0.0,0.5,0.1))
plt.title('long term average '+sound_name)
plt.show()
#####################################find F####################################
Fg=np.argmax(npcount)
mi=Fg-50
if(mi<0):
	mi=50
ma=Fg+50
if(ma>100):
	ma=100
M=0
for F in range(mi,ma):
	M+=(Fg-F)**2 * gf[F]
print M
	


	



###########################################################################semitone stability in all the graphs 
'''
for i in range(len(peaks)):
	if(peaks[i]>50):
		peaks[i]=100-peaks[i]
		
peaks.append(100)
peaks.append(0)
plt.plot(range(len(peaks)),peaks,'b.')
plt.xlabel('frames')
plt.ylabel('Pg(F,t)')
plt.title('semitone stability '+sound_name)
plt.show()
'''
#####################################################################
'''
data_arr=peaks




count_ar=[]
for i in range(100):
	count_ar.append(0)


lta=LTA.get_count(data_arr,count_ar)
plt.plot(lta,range(100))
plt.xlabel('g(F)')
plt.ylabel('cents')
plt.show()
'''
#######################################################################

