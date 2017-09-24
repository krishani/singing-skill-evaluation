
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
import sys
from scipy import stats


cent_vals=[]
Frange=100
x1=3000
x2=6000
pxf=[[(-1) for y in range(Frange)]for x in range(x1,x2)]

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
#sound_name='31082017035036.wav'
sound_name=str(sys.argv[1])
print sound_name
file_name=directory+sound_name
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

'''
plt.plot(t,pitches_with_no_zero)
plt.xlabel('time')
plt.ylabel('frequency hz')
plt.show()
'''
hz_to_cents.convert_to_cents(pitches_with_no_zero,cent_vals)
#cut frequencies below 3000 cents
cents=np.asarray(cent_vals, dtype=float)
t=np.arange(0,len(cents))

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
'''
###########################
start_time=time.time()

peaks=[]

num_of_points=len(cents)
frame_size=200

ss_frame=[]

	
	

num_of_frames=0
#########################################################################################
'''
for i in range(0,num_of_points-frame_size,50):

	F_val=[]
	frame=cents[i:frame_size+i]
	#print 'frameeeeeeeeeee start ------->'+str(i)

	for F in range(100):
		
		sigma=0
		for x in range(len(frame)):
			y=combfilter.gaussianComb(frame[x],F)
			sigma+=y
			#print 'x->'+ str(frame[x])+'F->'+str(F)+' gaussian value x,f->'+str(y)
	
		F_val.append(sigma)
		
	for k in range(len(F_val)):
		count_ar[k]+=F_val[k]
	
	npFval=np.asarray(F_val,dtype=float)
	#peak=np.argmax(npFval)
	print 'frame start'+str(i)+' end->'+str(frame_size+i)
	#peaks.append(peak)

	num_of_frames+=1
'''
########################################################################
def last_frame(frame):
	for x in range(len(frame)):
		sample=int(frame[x])
		#print 'sample'+str(sample)
		for F in range(100):
			
			
			if(x1<=sample<x2):
				#print 'within the range'
				#print 'F->'+str(F)
				#print pxf[sample-x1][F]
				#print pxf[3196-x1][0]
				
				if(pxf[sample-x1][F]==-1):
				
					y=combfilter.gaussianComb(sample,F)
					pxf[sample-x1][F]=y
				else:
					y=pxf[sample-x1][F]
				
			else:
				y=combfilter.gaussianComb(sample,F)
			
			
			#F_val[F]+=y
			count_ar[F]+=y
def update_array2(frame,half_ar):
	for x in range(len(frame)):
		sample=int(frame[x])
		#print 'sample'+str(sample)
		for F in range(100):
			
			
			if(x1<=sample<x2):
				#print 'within the range'
				#print 'F->'+str(F)
				#print pxf[sample-x1][F]
				#print pxf[3196-x1][0]
				
				if(pxf[sample-x1][F]==-1):
				
					y=combfilter.gaussianComb(sample,F)
					pxf[sample-x1][F]=y
				else:
					y=pxf[sample-x1][F]
				
			else:
				y=combfilter.gaussianComb(sample,F)
			
			
			#F_val[F]+=y
			count_ar[F]+=y
			half_ar[F]+=y
	

def update_array(x,half_ar):
	sample=int(frame[x])
	for F in range(100):
			
			
			if(x1<=sample<x2):
				#print 'within the range'
				#print 'F->'+str(F)
				#print pxf[sample-x1][F]
				#print pxf[3196-x1][0]
				
				if(pxf[sample-x1][F]==-1):
				
					y=combfilter.gaussianComb(sample,F)
					pxf[sample-x1][F]=y
				else:
					y=pxf[sample-x1][F]
				
			else:
				y=combfilter.gaussianComb(sample,F)
			
			
			#F_val[F]+=y
			count_ar[F]+=y
			half_ar[F]+=y
	
incr=50

count_ar=[]
for i in range(100):
	count_ar.append(0)
first_half=[0 for y in range(100)]
second_half=[0 for y in range(100)]
third_half=[0 for y in range(100)]
fourth_half=[0 for y in range(100)]

for i in range(0,num_of_points-frame_size,incr):
	frame=cents[i:frame_size+i]
	length=len(frame)
	#print len(frame)
	if(length<frame_size):
		#compute as normal
		last_frame(frame)
	else:
		
		#use the previous values
		if(i==0):
			print 'i==0'
			#if this is the first frame
			temp1=frame[0:50]
			update_array2(temp1,first_half)
			temp2=frame[50:100]
			update_array2(temp2,second_half)
			temp3=frame[100:150]
			update_array2(temp3,third_half)
			temp4=frame[150:200]
			update_array2(temp4,fourth_half)
			print ' ###############################################3frame number is -> '+str(num_of_frames)
			
			'''
			for x in range(len(frame)):
				if(0<=x<50):
					#first half
					update_array(x,first_half)
				elif(50<=x<100):
					#second half
					update_array(x,second_half)
				elif(100<=x<150):
					#third half
					update_array(x,third_half)
				elif(150<=x<200):
					#fourth half
					update_array(x,fourth_half)
			'''
		else:
			#update the count_ar
			for j in range(100):
				count_ar[j]+=second_half[j]+third_half[j]+fourth_half[j]
				
				#now update the 1st,2nd and 3rd ars
			
	
			first_half=second_half
			second_half=third_half
			third_half=fourth_half
			#compute the fourth half

			temp=frame[150:]
			
			fourth_half=[0 for y in range(100)]
			update_array2(temp,fourth_half)
			
					
	
	num_of_frames+=1
	if(num_of_frames==2):
		print'*************************************************************************************'
	
		
		for k in range(100):
				print ' value-> '+str(count_ar[k])+'1st->'+str(first_half[k]+second_half[k]+third_half[k]+fourth_half[k])			
	
	

				
'''
for i in range(0,num_of_points-frame_size,incr):
	F_val=np.zeros(100)
	sec_half=np.zeros(100)
	first_half=np.zeros(100)
	frame=cents[i:frame_size+i]
	finish=frame_size+i
	if(i==0):
		for x in range(len(frame)):
			for F in range(100):
				y=combfilter.gaussianComb(frame[x],F)
				F_val[F]+=y
				count_ar[F]+=y
				if(x>=incr):
					sec_half[F]+=y
				else:
					first_half[F]+=y
		for j in range(100):
			print 'FULL-> '+str(count_ar[j])+' F_irst half-> '+str(first_half[j])+' second_half->'+str(sec_half[j])
	else:
		print'under construction'

'''									
			
	



########################################################################

print num_of_frames
npcount=np.asarray(count_ar,dtype=float)
gf=npcount/num_of_frames
print "My program took", time.time() - start_time, "to run"


plt.plot(gf,range(100))
plt.xlabel('g(F)')
plt.ylabel('cent')
plt.xticks(np.arange(0.0,0.5,0.1))
plt.title('long term average '+sound_name)
plt.show()
#####################################find F####################################
Fg=np.argmax(npcount)
mi=Fg-50
if(mi<0):
	mi=0
ma=Fg+50
if(ma>100):
	ma=100
M=0
for F in range(mi,ma):
	M+=(Fg-F)**2 * gf[F]
#################################### FEATURE 1 IS M
print 'M value->'+str(M)
###################################################################################
def GF(Fg):
	GF_val=[0 for i in range(50)]
	for f in range(0,50):
		plus=Fg+f
		minus=Fg-f
		if(plus>=100):
			plus=99
		if(minus<0):
			minus=0
		avg=(gf[plus]+gf[minus])/2.0
		GF_val[f]=avg
	return GF_val


y=GF(Fg)
x=np.arange(0,len(y))
slope,intercept,r_value,p_value,std_err=stats.linregress(x,y)

print "slope-> "+str(slope)+' M value->'+str(M)

plt.plot(x,y, 'o', label='average G(F)')
plt.plot(x,intercept + slope*x,'r', label='fitted line')
plt.legend()
plt.title("average G(F) "+sound_name)
plt.show()


####################################################### FEATURE 2 is slope
	



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

