from subprocess import Popen, PIPE, call

from collections import Counter
import numpy as np
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

def getHop(tempo,timeSig):
    if timeSig == '4/4':
        hop = (60.0/tempo)*4*100
    elif timeSig == '3/4':
        hop = (60.0/tempo)*3*100

    return int(round(hop))


def sliceMeasures(pitchlist,tempo,timeSig):
    hop = getHop(tempo,timeSig)
    i = 0
    song = []
    while True:

        if i + hop <= len(pitchlist):
            song.append(pitchlist[i:i+hop])
        else:
            song.append(pitchlist[i:])
            break

        i += hop

    return song


def getContinuosPitchClass(f):
    if f is not 0:
        return 12*math.log(((f+0.0)/523.2),2)
    else:
        return 0

def alignPitch(measure):
    totpoffset=0
    for i in range(len(measure)):
        if measure[i] is not 0:
            p = getContinuosPitchClass(measure[i])
            measure[i] = ps
            totpoffset += round(p) - p
    measure = [x for x in measure if x != 0]
    if len(measure) is not 0:
        #meanpoffset = totpoffset/len(measure)
        meanpoffset = 0
        for j in range(len(measure)):
            measure[j] = int(round(measure[j] + meanpoffset)) % 12

        return measure
    else:
        return []

def viterbi_alg(A_mat, O_mat, observations):
    # get number of states
    num_obs = observations.size
    num_states = A_mat.shape[0]
    # initialize path costs going into each state, start with 0
    log_probs = np.zeros(num_states)
    # initialize arrays to store best paths, 1 row for each ending state
    paths = np.zeros( (num_states, num_obs+1 ))
    paths[:, 0] = np.arange(num_states)
    # start looping
    for obs_ind, obs_val in enumerate(observations):
        # for each obs, need to check for best path into each state
        for state_ind in xrange(num_states):
            # given observation, check prob of each path
            temp_probs = log_probs + \
                         np.log(O_mat[state_ind, obs_val]) + \
                         np.log(A_mat[:, state_ind])
            # check for largest score
            #print "temp ",temp_probs
            best_temp_ind = np.argmax(temp_probs)
            # save the path with a higher prob and score
            paths[state_ind,:] = paths[best_temp_ind,:]
            paths[state_ind,(obs_ind+1)] = state_ind
            log_probs[state_ind] = temp_probs[best_temp_ind]
    # we now have a best stuff going into each path, find the best score
    best_path_ind = np.argmax(log_probs)
    # done, get out.
    #return (best_path_ind, paths, log_probs)
    return (best_path_ind, paths)


def viterbi(obs, states, start_p, trans_p, emit_p):
    V = [{}]
    for st in states:
        V[0][st] = {"prob": start_p[st] * emit_p[st][obs[0]], "prev": None}
    # Run Viterbi when t > 0
    for t in range(1, len(obs)):
        V.append({})
        for st in states:
            max_tr_prob = max(V[t - 1][prev_st]["prob"] * trans_p[prev_st][st] for prev_st in states)
            for prev_st in states:
                if V[t - 1][prev_st]["prob"] * trans_p[prev_st][st] == max_tr_prob:
                    max_prob = max_tr_prob * emit_p[st][obs[t]]
                    V[t][st] = {"prob": max_prob, "prev": prev_st}
                    break
    for line in dptable(V):
        print line
    opt = []
    # The highest probability
    max_prob = max(value["prob"] for value in V[-1].values())
    previous = None
    # Get most probable state and its backtrack
    for st, data in V[-1].items():
        if data["prob"] == max_prob:
            opt.append(st)
            previous = st
            break
    # Follow the backtrack till the first observation
    for t in range(len(V) - 2, -1, -1):
        opt.insert(0, V[t + 1][previous]["prev"])
        previous = V[t + 1][previous]["prev"]

    print 'The steps of states are ' + ' '.join(opt) + ' with highest probability of %s' % max_prob


def dptable(V):
    # Print a table of steps from dictionary
    yield " ".join(("%12d" % i) for i in range(len(V)))
    for state in V[0]:
        yield "%.7s: " % state + " ".join("%.7s" % ("%f" % v[state]["prob"]) for v in V)

obs = ('normal', 'cold', 'dizzy')
states = ('Healthy', 'Fever')
start_p = {'Healthy': 0.6, 'Fever': 0.4}
trans_p = {
   'Healthy' : {'Healthy': 0.7, 'Fever': 0.3},
   'Fever' : {'Healthy': 0.4, 'Fever': 0.6}
   }
emit_p = {
   'Healthy' : {'normal': 0.5, 'cold': 0.4, 'dizzy': 0.1},
   'Fever' : {'normal': 0.1, 'cold': 0.3, 'dizzy': 0.6}
   }

text = toPitch('nilnew.mp3')
pitchDic,timelist,pitchlist = read_praat_out(text)

song = sliceMeasures(pitchlist,130,'3/4')

song = [alignPitch(measure) for measure in song]
finalChords=[]

A_mat = np.load('ctmatrix.npy')
O_mat = np.load('comatrix.npy')
for observations in song:
    obs = np.array(observations)
    best_path_ind, paths = viterbi_alg(A_mat,O_mat,obs)
    #print "obs1, best path is" + str(paths[best_path_ind,:])
    finalChords.append(paths[best_path_ind,:])

out = []

for i in finalChords:
    #todo : i could be empty
    b = Counter(i)
    c = b.most_common(1)
    print c
    out.append(reverseChordIndex.get(int(c[0][0])))
