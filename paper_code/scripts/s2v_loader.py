import numpy as np
import tqdm
import pickle as pkl

final_dict = {}
#s2v_txt = '/scratch/prakhar/s2vmodels_twitter_models/s2v_model_300_2gram_ep1_5M_mc20.txt'
s2v_txt = "/scratch/prakhar/s2vmodels_twitter_models/s2v_model_700_2gram_ep1_5Mvocab.txt"
s2v_out = '/scratch/hartley/twitter_covid_insights/insights_All/s2v_dict_700.pkl'
with open(s2v_txt, encoding='utf8') as f :
    lines = f.readlines()
    for e, line in tqdm.tqdm(enumerate(lines), total=len(lines)) :
        split = line.split(' ')
        word = split[0]
        if word[0] == '#' :
            #print('split', split[1:])
            if e == len(lines) - 1 :
                arr = np.array(split[1:]).astype(float)
                if len(arr) != 300 : 
                    raise Exception (f'len not 300 : {split[1:]}')
            else :
                # Account for final \n
                arr = np.array(split[1:-1]).astype(float)
                
                if len(arr) != 700 : 
                    raise Exception (f'len not 300 : {split[1:-1]}')
            
            
                
            #print('len arr:', len(arr))
            final_dict[word] = arr
pkl.dump(final_dict, open(s2v_out, "wb"), protocol=4)