'''
This code reads all the moments from FLASK simulations;
it performs the noise subtraction (FLASK moments haven0t been de-noised)
and if needed add the 'modelling uncertainties'. Then it saves a list of corrected
FLASK moments objects - this is what is needed to run the analysis.

'''

'''
de-noise.
Based on FLASK:
only auto-moments for <dd> ans <ddd>.
<dd> : <nn> needs to be subtracted, <nd>  2order smaller
<ddd> : <nnn> needs to be subtracted, <ndd>, <nnd>  2order smaller

<dK> : need to subtract <dKN> , others zero.
<dKK>: need to subtract everything, dkNkN dominant
<KKd>: need to subtract everything
'''

import copy
import os
import pickle
from Moments_analysis import moments_map, ∑

def save_obj(name, obj):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        
def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)#, encoding='latin1')
    
output_FLASK = '/global/cscratch1/sd/mgatti/Mass_Mapping/moments/flask/'
N_rel = 500
mapp = []
bins_lenses = [0, 1, 2, 3]
bins_WL = [0, 1, 2, 3]
# dict_keys(['kEkE', 'kBkB', 'kEkN', 'kNkN', 'kNkE', 'dKK', 'Kdd', 'dkNkN', 'nKK', 'nkNkN', 'kNdd', 'knn', 'kNnn', 'dd', 'nn', 'dnn', 'ndd'])
 
for i in range(N_rel):
    try:
        mute = load_obj(output_FLASK+'moments_seed_'+str(i+1))
        mapp.append(copy.deepcopy(mute))
        if i == 0 :
            mapp_ave = copy.copy(mute)
            for key in mapp_ave.moments.keys():
                for key2 in mapp_ave.moments[key].keys():
                    mapp_ave.moments[key][key2] = mapp_ave.moments[key][key2]/N_rel
        else:
            for key in mapp_ave.moments.keys():
                for key2 in mapp_ave.moments[key].keys():
                    mapp_ave.moments[key][key2] += mute.moments[key][key2]/N_rel
         


        for b1 in bins_WL:
            # noise subtraction
            binx = '{0}_{0}'.format(b1)

            mapp[i].moments['kEkE'][binx] -= mapp[i-1].moments['kNkN'][binx]
            mapp[i].moments['kBkB'][binx] -= mapp[i-1].moments['kNkN'][binx]

        for b1 in bins_lenses:
            for b2 in bins_WL:
                binx = '{0}_{1}'.format(b1,b2)
                mapp[i].moments['dKK'][binx] -= mapp[i-1].moments['dkNkN'][binx]
                binx = '{0}_{1}'.format(b2,b1)
                mapp[i].moments['Kdd'][binx] -= mapp[i-1].moments['kNdd'][binx]


        for b1 in bins_lenses:
            for b2 in bins_WL:
                for b3 in bins_WL:
                    try:
                        binx = '{0}_{1}_{2}'.format(b1, b2, b3)
                        mapp[i].moments['dKK'][binx] -= (mapp[i-1].moments['nkNkN'][binx]+3*mapp[i-1].moments['nKK'][binx]+3*mapp[i-1].moments['dkNkN'][binx])
                    except:
                        pass
        for b1 in bins_WL:
            for b2 in bins_lenses:
                for b3 in bins_lenses:
                    try:
                        binx = '{0}_{1}_{2}'.format(b1, b2, b3)
                        mapp[i].moments['Kdd'][binx] -= (mapp[i-1].moments['kNnn'][binx]+3*mapp[i-1].moments['kNdd'][binx]+3*mapp[i-1].moments['knn'][binx])
                    except:
                        pass

    except:
        print ('failed ',i)
        mapp.append(mapp[i-1])
        for key in mapp_ave.moments.keys():
            for key2 in mapp_ave.moments[key].keys():
                mapp_ave.moments[key][key2] += mapp[i-1].moments[key][key2]/N_rel
save_obj('/global/project/projectdirs/des/mgatti/Moments_analysis/Cov_FLASK_Y3',mapp)
print ('done')