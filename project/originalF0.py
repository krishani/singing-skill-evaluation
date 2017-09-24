from subprocess import Popen, PIPE, call
from scipy import linspace, polyval, polyfit, sqrt, stats, randn,signal
from collections import Counter
from sklearn import datasets, linear_model
import numpy as np
import matplotlib.pyplot as plt
import math
import csv
import switch
import approximate
import hz_to_cents
import cante
import LTA
import combfilter



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
file_name=directory+'sarigama.wav'
text = toPitch(file_name)
pitchDic,timelist,pitchlist = read_praat_out(text)






times=[]
durations=[]
notes=[]
cent_vals=[]
offset_vals=[]
frame_vals=[]
diffs=[]

pitch_values=[]
def get_the_key(note_index,time_start):
	x=(note_index-time_start)//0.01
	ind=(x*0.01)+time_start
	print ind
	return ind

def get_the_mean(ar):
	x=0
	sum=0.0
	for pitch in ar:
		if(pitch!=0.0):
			x+=1
		sum+=pitch
	if(x==0):
		mean=0
	else:

		mean=float(sum)/x
	return mean


#start-cante start of the note
#end-end of the note
#time_start=timelist[0]

def get_the_gap(start,end,time_start,ar,real_note):
	s=get_the_key(start,time_start)
	e=get_the_key(end,time_start)
	
		
	pre_hz_value=0.0
	
	
	hz_value=switch.mnotes(real_note)
	post_hz_value=0.0
	
	
	x=s
	temp=[]
	while(x<=e):
		
		x=round(x,3)
		#print 'pitch val'
		y= pitchDic[x]
		
		#print y['Pitch']
		temp.append(y['Pitch'])
		#ar.append(y['Pitch'])
		pit=y['Pitch']
		ar.append(pit)
		#if(pit!=0):
		#	dif=abs(pit-hz_value)
		#	ar.append(dif)
		

			
		
			
		
		x+=0.01
	#print 'temp'
	print temp
	#print 'mean'

	###temp_np=np.asarray(temp,dtype=float)
	#m=np.median(temp_np)
	###m=get_the_mean(temp)
	#ar.append(m)
	
	'''
	dif=approximate.check_nearest_note(real_note,m)
	ar.append(dif)
	##############################################
	print 'start_ind'+" "+str(s)+" end_index "+str(e)+" meadian "+str(m)+" note_val "+str(hz_value)+" difference"+str(dif)
	if(dif>=50.0):
		print "problem&&&&&&&&&&&&&&&&&&&&&&&&&&"
	'''
	return ar
x=timelist[0]
time_start=float(timelist[0])

cante.transcribe(file_name, acc=True, f0_file=False, recursive=False)

base,ext=file_name.split(".")
new_part='.notes.csv'
new_fname=base+new_part



with open(new_fname,'rb')as f:
	reader=csv.reader(f)
	for row in reader:
		
		times.append(round(float(row[0]),3))
		durations.append(round(float(row[1]),3))
		notes.append(row[2])
#get_the_gap(4.362,,time_start,pitch_values)
for i in range(len(times)):
	get_the_gap(times[i],times[i]+durations[i],time_start,diffs,int(notes[i]))
diffs_normal=[]
hz_to_cents.convert_to_cents(diffs,diffs_normal)

#diffs.append(100)
'''
for i in range(len(diffs)):
	if(diffs[i]<=100):
		 diffs_normal.append(diffs[i])
'''		,
diff_np=np.asarray(diffs_normal, dtype=float)


time_np=np.arange(0,len(diff_np),1)
plt.figure
plt.plot(time_np,diff_np, 'b')
plt.xlabel('note')
plt.ylabel('FO CENTS')
plt.title(base)
plt.show()

offsets=[]
hz_to_cents.get_offset(diff_np,offsets)
#offsets.append(100)
off_np=np.asarray(offsets, dtype=float)
time_np1=np.arange(0,len(off_np),1)

plt.figure
plt.plot(time_np1,off_np, 'b.')
plt.xlabel('note')
plt.ylabel('offset CENTS')
plt.title(base + "  OFFSETS")
plt.show()

frame_vals=[]
hz_to_cents.break_into_frames2(offsets,50,frame_vals)
frame_vals.append(100)

frame_np=np.asarray(frame_vals, dtype=float)
time_np2=np.arange(0,len(frame_np),1)

plt.figure
plt.plot(time_np2,frame_np, 'b.')
plt.xlabel('note')
plt.ylabel('offset CENTS mean')
plt.title(base+ "  mean offsets")
plt.show()
############################################LTA
data_arr=offsets
frame_size=10
frame_arr=[]

LTA.break_to_frames(data_arr,10,frame_arr)
count_ar=[]
for i in range(100):
	count_ar.append(0)


lta=LTA.get_the_frequency(frame_arr,count_ar)
ltatime=np.arange(0,len(lta),1)

plt.figure
plt.plot(ltatime,lta)
plt.ylabel('g(F)')
plt.xlabel('cents')
plt.title('LTA')
plt.show()
	


#cents=[]
#hz_to_cents.convert_to_cents(diffs,cents)
#cents.append(100)
#cents_np=np.asarray(cents, dtype=float)

'''
for i in range(len(times)):
	get_the_gap(times[i],times[i]+durations[i],time_start,pitch_values)

print pitch_values

pitches=np.asarray(pitch_values, dtype=float)
#remove silent sections
index = np.argwhere(pitches==0.0)
#print index
pitches_with_no_zero = np.delete(pitches, index)

	
cents_original=np.asarray(pitches_with_no_zero, dtype=float)

time_original=np.arange(0,len(cents_original),1)

plt.figure
plt.plot(time_original, cents_original, 'b')
plt.show()
	
def convert_to_cents(pitchlist):
#this function will take the original f0 values and converts it to cents
	for pitch in pitchlist:
		if(pitch==0.0):
			x=pitch
		else:
			
			x=int(round(1200*math.log(pitch/(440*math.pow(2,3/12-5)),2)))
		
		cent_vals.append(x)
	#print cent_vals
	return cent_vals

################################################################
convert_to_cents(cents_original)

def get_offset(cent_vals):
	for pitch in cent_vals:
		x=pitch%100
		if(x>50):
			offset=100-x
		else:
			offset=x
			
		
		offset_vals.append(offset)
	#print offset_vals
	return offset_vals
######################################################################

get_offset(cent_vals)
offsets=np.asarray(offset_vals, dtype=float)

offset_time=np.arange(0,len(offsets),1)
plt.figure
plt.plot(offset_time,offsets, 'b.')
plt.show()
	
'''
