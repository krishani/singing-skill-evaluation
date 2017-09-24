def get_the_key(note_index,time_start):
	x=(note_index-time_start)//0.01
	ind=(x*0.01)+time_start
	print ind
	return ind
get_the_key(4.104,0.021)#4.101
get_the_key(4.362,0.021)#4.361
get_the_key(12.164,0.021)#12.161
get_the_key(12.402,0.021)#12.401
get_the_key(12.934,0.021)#12.931
get_the_key(14.167,0.021)#14.161 #here it is more close to 14.171

def get_the_mean(ar):
	n=len(ar)
	sum=0
	for pitch in ar:
		sum+=pitch
	mean=sum/n
	return mean
		
