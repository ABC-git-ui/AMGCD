import numpy as np
import torch
import time
import os
from train import Train
from args import args
from utils import setup_seed
import warnings
import pandas as pd
warnings.filterwarnings('ignore')
if __name__ == '__main__':
    if not os.path.exists('.checkpoints'):
        os.makedirs('.checkpoints')
    setup_seed(args.seed, torch.cuda.is_available())


    acc_list = []
    rec_list = []
    prec_list = []
    f1_list = []
    auc_list = []
    aupr_list = []

    ftr_list = []
    tpr_list = []
    r_list = []
    p_list = []

    repeats = 5
    for repeat in range(repeats):
        print('-------------------- Repeat {} Start -------------------'.format(repeat))

        train = Train(args,repeat,acc_list,rec_list,prec_list,f1_list,auc_list,aupr_list,ftr_list,tpr_list,r_list,p_list)
        t_total = time.time()

        for epoch in range(args.epoch_pre):
            train.pre_train_truck(epoch)
        train.save_checkpoint(ts='teacher_truck')

    
        for epoch in range(args.epoch_pre):
            train.pre_train_leaf(epoch)
        train.save_checkpoint(ts='teacher_sub')

        # load best pre-train teahcer models
        train.load_checkpoint(ts='teacher_truck')
        train.load_checkpoint(ts='teacher_sub')
        print('\n--------------\n')

        # train student model GCN
        for epoch in range(args.epoch_stu):
            train.train_student(epoch)
        train.save_checkpoint(ts='student')

      
        # test student model GCN
        train.load_checkpoint(ts='student')
        train.test('student')

        print('******************** Repeat {} Done ********************\n'.format(repeat+1))


    
    df = pd.DataFrame()
    df['auc'] = auc_list
    df['aupr'] = aupr_list
    df['acc'] = acc_list
    df['rec'] = rec_list
    df['pre'] = prec_list
    df['f1'] = f1_list
    df.to_csv('./result/result.csv',index=0)
    pd.DataFrame(ftr_list).to_csv('./result/fpr.csv',index=0,header=0)
    pd.DataFrame(tpr_list).to_csv('./result/tpr.csv',index=0,header=0)
    pd.DataFrame(r_list).to_csv('./result/recall.csv',index=0,header=0)
    pd.DataFrame(p_list).to_csv('./result/precision.csv',index=0,header=0)
        

    print(len(tpr_list))
    print('Result auc: {}'.format(auc_list))
    print('Result aupr: {}'.format(aupr_list))
    print('Result acc: {}'.format(acc_list))
    print('Result re: {}'.format(rec_list))
    print('Result pre: {}'.format(prec_list))
    print('Result f1: {}'.format(f1_list))

    print('Avg AUC: {:.6f}'.format(sum(auc_list) / repeats))
    print('Avg AUPR: {:.6f}'.format(sum(aupr_list) / repeats))    
    print('\nAll Done!')



