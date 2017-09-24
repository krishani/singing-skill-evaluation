import numpy as np
import switch
import hz_to_cents
max=95
def check_nearest_note(real_note,meanv):
	

	hz_vals=[0.0,0.0,0.0]
	diff_vals=[0.0,0.0,0.0]
	hz_vals[1]=switch.mnotes(real_note)
	



	hz_vals[1]=switch.mnotes(real_note)
	if(real_note==0):
		
		hz_vals[0]=0
		hz_vals[2]=switch.mnotes(real_note+1)
	elif(real_note==max):
		hz_vals[0]=switch.mnotes(real_note-1)
		hz_vals[2]=max
	else:
		hz_vals[0]=switch.mnotes(real_note-1)
		hz_vals[2]=switch.mnotes(real_note+1)
	#pre_note0 realnote1 post note 2 meanval3
	'''
	pitchlist=[]
	pitchlist.append(hz_vals[0])
	pitchlist.append(hz_vals[1])
	pitchlist.append(hz_vals[2])
	pitchlist.append(meanv)
	cent_vals=[]

	hz_to_cents.convert_to_cents(pitchlist,cent_vals)
	'''
	for i in range(len(hz_vals)):
		diff_vals[i]=abs(meanv-hz_vals[i])
		
	m=min_ind(diff_vals)
	
	
	return m
	
	

def min_ind(ar):
	ar_np=np.asarray(ar,dtype=float)
	m=np.amin(ar_np)
	ind=np.argwhere(ar_np==m)
	
	return m
#print(check_nearest_note(71,467.625))
#print(check_nearest_note(69,467.18))

