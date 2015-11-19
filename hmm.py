__author__ = 'shranith'
import numpy as np
import math
from sklearn.preprocessing import normalize

obs = []
states = []

f = open('entrain.txt','r')
eachline = f.readline()
while eachline != '':
    items = eachline.split('/')
    obs.append(items[0])
    states.append(items[1])
    eachline = f.readline()
#print(obs)

tot_states = len(set(states))
total_obs = len(obs)


ids = 0
obs_id_list = {'OOV':ids}#OOV out of vocabulary
ids = ids + 1
for i in range(total_obs):
    if obs[i] not in obs_id_list:
        obs_id_list[obs[i]] = ids
        ids += 1
        obs[ids] = 'OOV'
'''state_ids Map maintianing Ids of each and every tag with key being the tag.state_ids_rev is a Map with key being ID and value being tag'''
#print len(set(obs))
#print
state_ids={}
state_ids_rev = {}
ids = 0
for i in range(len(states)):
    if states[i].strip() not in state_ids:
        state_ids[str(states[i].strip())] = ids
        state_ids_rev[ids] = str(states[i].strip())
        ids += 1
#print states
#print state_ids_rev[0]
first_counts = {}
for i in set(states):
    first_counts[state_ids[i.strip()]] = 0

#print len(obs_id_list)
for i in range(len(states)-1):
    if state_ids[states[i].strip()] == 0:
        first_counts[state_ids[states[i+1].strip()]]+=1

'''Giving each and evry unique tag a unique ID in the for loop'''

obs_prob = np.ones((tot_states,len(obs_id_list)), dtype=float)
state_trans = np.ones((tot_states,tot_states), dtype=float)
#print obs_prob
#print state_trans
''' Intializing the transition prob matrix and obs probabiltie matrix with value 1 '''

for i in range(len(states)-1):
    x = state_ids[states[i].strip()]
    y = obs_id_list[obs[i]]
    y1 = state_ids[states[i+1].strip()]
    obs_prob[x,y] += 1
    state_trans[x,y1] += 1

x = state_ids[states[i].strip()]
y = obs_id_list[obs[i]]

obs_prob[x,y] += 1

''' Counting the values for each and every state transition and observation state pair'''

obs_prob = normalize(obs_prob, norm='l1', axis=1)
state_trans = normalize(state_trans, norm='l1', axis=1)

'''Finding the probability of the transition matrix and observation matrix using the normalize function '''



test_obs = []
test_states = []

f = open('entest.txt','r')
row = f.readline()
while row!= '':
    items = row.split('/')
    test_obs.append(items[0])
    test_states.append(items[1].strip())
    row = f.readline()


for i in range(len(test_obs)):
    if test_obs[i] not in obs_id_list:
        test_obs[i] = 'OOV'


'''
# Use Viterbi algorithm for prediction of the tags.

# Using logs for the computation as multiplying probabailities gets smaller and smaller.
# Using log(p1 X p2) = log(p1) + log(p2) this we can solve the processing.
'''
maximum_path = [] #Saves the tag with the maximum value till now to the current observation

tag = obs_id_list[test_obs[0]] #storing the ID of the first observation
''' #As we have saved our transition probabilities and observation probabilities in terms of IDS
#  we get Ids back and store in k'''

maximum_state = [math.log(((1.0 + first_counts[i])/(tot_states + len(set(obs))))*obs_prob[i,tag],2) for i in range(tot_states)]
'''#maximum_state  contains the list of all the possible states probabilites
'''

for i in range(1,len(test_obs)): # Now we go through eah and every obervation
	maxi = np.argmax(maximum_state)
	maximum_path.append(state_ids_rev[maxi])
	current_maximum_state = []

	for x in range(tot_states): #We calculate the probability of all the states from the current state ad multiply with correspoding observation to find the maximum value
		temp = []
		k = obs_id_list[test_obs[i]]
		for y in range(tot_states):
			temp.append(maximum_state[y]+math.log(state_trans[y,x],2)+math.log(obs_prob[x,k],2))
		current_maximum_state.append(max(temp))
	maximum_state = current_maximum_state

maxi = np.argmax(maximum_state)
maximum_path.append(state_ids_rev[maxi])



# Calculate Error rate.
err = 0
for i in range(0,len(test_obs)):
	if test_states[i] != maximum_path[i]:
		err+=1
err = (err)*1.0/len(test_obs)


print "Error rate = ", float('%0.4f'%(err*100)), "%"
