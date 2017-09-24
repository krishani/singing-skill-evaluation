import math
import numpy as np

def convert_to_cents(pitchlist,cent_vals):
#this function will take the original f0 values and converts it to cents
	for pitch in pitchlist:
		if(pitch==0.0):
			x=pitch
		else:
			
			x=int(round(1200*math.log(pitch/(440*math.pow(2,3/12-5)),2)))
		
		cent_vals.append(x)
	#print cent_vals
	return cent_vals
#print convert_to_cents([0,50,60,70,80,90,100,150,200],[])
def to_cents(pitchlist,cent_vals):
	for pitch in pitchlist:
		if(pitch==0.0):
			x=pitch
		else:
			
			x=int(round(1200*math.log(pitch/(440*math.pow(2,3/12-5)),2)))
		if(x>3000):

		
			cent_vals.append(x)
	#print cent_vals
	return cent_vals
	
'''
def get_offset(cent_vals,offset_vals):
	for pitch in cent_vals:
		x=pitch%100
		if(x>50):
			offset=100-x
		else:
			offset=x
			
		
		offset_vals.append(offset)
	#print offset_vals
	return offset_vals

def break_into_frames2(pitch_ar,size,frame_vals):
	#t=np.arange(0,len(pitch_ar),size)
	#print len(t)
	
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
'''
