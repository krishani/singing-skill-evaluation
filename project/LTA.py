import numpy as np
import matplotlib 


def break_to_frames(data_arr,frame_size,frame_arr):
	n=len(data_arr)
	frame_num=0
	rem=n%frame_size
	if(rem==0):
		frame_num=n//frame_size
	else:
		frame_num=n//frame_size +1
	print 'num of frames is'+str(frame_num)
	j=1
	for i in range(0,len(data_arr),frame_size):
		
		print str(i)+ 'i'
		
		temp=data_arr[i:j*frame_size]
		print 'temp'
		print temp

		frame_arr.append(temp)
		j+=1
		
	print frame_arr
	return frame_arr	
	
'''
x=[]
for i in range(100):
	x.append(1)
frame_size=10
frame_ar=[]
break_to_frames(x,10,frame_ar)	
#data array input must be a normal array
'''
def get_the_frequency(frame_ar,count_ar):
	for ar in frame_ar:
		
		
		for i in ar:
			
			
			num_round=int(round(i))
			print num_round
			print count_ar[num_round]
			count_ar[num_round]+=1
	#print count_ar
	avg=np.asarray(count_ar,dtype=float)
	avg=avg/10
	print avg
	return avg

def get_count(peaks,count_ar):
	for peak in peaks:
		count_ar[peak]+=1
	avg=np.asarray(count_ar,dtype=float)
	avg=avg/100
	print avg
	return avg
	
'''
count_ar=[]
for i in range(100):
	count_ar.append(0)


print(get_the_frequency(frame_ar,count_ar))
'''
'''
			
x=np.ones(100)
frame_size=10
frame_arr=[]
count_ar=np.zeros(100)
break_to_frames(x,10,frame_arr)	
#get_the_frequency(frame_arr,count_ar)	
	
	
	
'''
'''
np.ones(100)
frame_size=10

ar1=[2,6,2,0,0,0,0,0,0,0,0]
ar2=[1,6,3,0,0,0,0,0,0,0,0]
ar3=[1,6,3,0,0,0,0,0,0,0,0]
ar4=[1,6,2,1,0,0,0,0,0,0,0]
ar5=[1,5,2,2,0,0,0,0,0,0,0]
ar6=[1,6,3,0,0,0,0,0,0,0,0]
ar7=[1,6,3,0,0,0,0,0,0,0,0]
ar8=[0,7,3,0,0,0,0,0,0,0,0]
ar9=[0,7,3,0,0,0,0,0,0,0,0]
ar10=[0,7,3,0,0,0,0,0,0,0,0]
LTM_AR=[0,0,0,0,0,0,0,0,0,0]
no_fr=10

for i in range(10):
	LTM_AR[i]=(ar1[i]+ar2[i]+ar3[i]+ar4[i]+ar5[i]+ar6[i]+ar7[i]+ar8[i]+ar9[i]+ar10[i])
	print LTM_AR[i]
ltm_np=np.asarray(LTM_AR,dtype=float)
ltm_np=ltm_np/10
print ltm_np

'''	
