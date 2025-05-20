import numpy as np
from tqdm import tqdm

import configs
from functions.hashing import get_hamm_dist
from functions.metrics import preprocess_for_calculate_mAP
from scipy.spatial.distance import cdist
from sklearn.metrics import auc
import matplotlib.pyplot as plt

draw_range = [1, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500,
              9000, 9500, 10000]




def collision_rate(data_set):
    data_set = data_set.cpu().numpy().tolist()
    hash_values = set()
    collisions = 0
    total_samples = len(data_set)

    for data in data_set:
        hash_value = ",".join([str(int(x)) for x in data])
        if hash_value in hash_values:
            collisions += 1
        hash_values.add(hash_value)

    collision_rate = collisions / total_samples
    return collision_rate

def pr_curve(qF, rF, qL, rL, what=1, topK=-1):

    rF, qF = preprocess_for_calculate_mAP(rF, qF, ternarization=None, 
                                distance_func='hamming', zero_mean=False)

    print("--test collision_rate ", collision_rate(qF))
    print("--db collision_rate ", collision_rate(rF))

    qF = qF.cpu().numpy()
    rF = rF.cpu().numpy()
    qL = qL.cpu().numpy()
    rL = rL.cpu().numpy()

    

    n_query = qF.shape[0]
    if topK == -1 or topK > rF.shape[0]: 
        topK = rF.shape[0]
        topks = [1, 50, 100, 200, 300, 400, 500, 600, 
            800, 1000, 1200, 1600, 2000, 3000, 4000, 
            5000, 7000, 10000, 15000, 20000, 25000, 32322]
        save_name = 'AUC_all.png'
    else:
        topks = [k for k in range(1, topK + 1, (topK + 1)//21)]
        save_name = 'AUC_topk.png'


    Gnd = (np.dot(qL, rL.T) > 0).astype(np.float32) #n_query x n_database

    if what == 0:
        Rank = np.argsort(cdist(qF, rF, 'cosine'))
    else:
        Rank = np.argsort(cdist(qF, rF, 'hamming'))

    P, R = [], []
    for k in topks:

        p = np.zeros(n_query)  
        r = np.zeros(n_query)  
        for it in range(n_query):  
            gnd = Gnd[it] 
            gnd_all = np.sum(gnd)  
            if gnd_all == 0:
                print("warning: no same label in db")
                continue
            asc_id = Rank[it][:k]
            gnd = gnd[asc_id]
   
            with open("badcase_output/%d_topk_ids.txt" % it, 'w') as f:
                for idx in asc_id:
                    f.write(str(idx)+'\n')
            with open("badcase_output/%d_topk_labels.txt" % it, 'w') as f:
                for idx in gnd:
                    f.write(str(idx)+'\n')
            gnd_r = np.sum(gnd)  

            p[it] = gnd_r / k 
            r[it] = gnd_r / gnd_all

        P.append(np.mean(p))
        R.append(np.mean(r))

    fig = plt.figure(figsize=(5, 5))
    plt.plot(R, P, marker='s')  
    plt.grid(True)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.legend()
    plt.savefig(save_name)
    
    print(P)
    print(R)


    pr_auc = auc(R,P)
    return pr_auc