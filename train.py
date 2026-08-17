from __future__ import division
from __future__ import print_function

import time
import torch
import torch.nn as nn
import torch.optim as optim
import scipy.sparse as sp
from utils import load_data, accuracy,auroc,auprc,prediction,setup_seed
from models import GCN, mlp,Net_truck,Sub_leafNet,FuseAttention
from args import args
from logit_losses import *
import pandas as pd
import numpy as np
# Model and optimizer
class Train:
    def __init__(self, args,repeat,acc_list,rec_list,prec_list,f1_list,auc_list,aupr_list,ftr_list,tpr_list,r_list,p_list):
        self.args = args
        self.repeat = repeat
        self.best_teacher_truck_val, self.best_teacher_sub_val, self.best_student_val = 0, 0, 0
        self.teacher_truck_state,  self.teacher_sub_state, self.student_state = None, None, None
        self.load_data()

        self.acc_list = acc_list
        self.rec_list = rec_list
        self.prec_list = prec_list
        self.f1_list = f1_list
        self.auc_list = auc_list
        self.aupr_list = aupr_list

        self.ftr_list = ftr_list
        self.tpr_list = tpr_list
        self.r_list = r_list
        self.p_list = p_list

        self.stu_model = GCN(nfeat=self.features.shape[1],
                           nhid=128,
                           nclass=128,
                           dropout=self.args.dropout,
                           nhid_feat=128,
                           nhid_stru=128)
        self.classify = mlp(256,128)
        self.classify2 = mlp(256,128)
        self.stu_model.to(args.device)
        self.classify.to(args.device)
        self.truck_model = Net_truck(self.features.shape[1],128,5)
        self.sub_leafNet = Sub_leafNet(self.features.shape[1], self.features.shape[0], 128, 128, 128, 0.1, args.device)
        self.fuse = FuseAttention()
        self.fuse.to(args.device)
        self.truck_model.to(args.device)
        self.sub_leafNet.to(args.device)
        self.classify2.to(args.device)
        # Setup loss criterion
        self.criterionTruck = torch.nn.BCELoss(reduction='mean')#nn.CrossEntropyLoss()
        self.criterionSub = torch.nn.BCELoss(reduction='mean')#nn.CrossEntropyLoss()
        self.criterionStudent = torch.nn.BCELoss(reduction='mean')#nn.CrossEntropyLoss()
  

        # Setup Training Optimizer
        self.optimizerTruck = optim.Adam(self.truck_model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.optimizerSub = optim.Adam(self.sub_leafNet.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.optimizerStudent = optim.Adam(self.stu_model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.optimizerMLP = optim.Adam(self.classify.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        self.optimizerMLP2 = optim.Adam(self.classify2.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
    def load_data(self):
        # load data

        setup_seed(args.seed, torch.cuda.is_available())

        self.tadj, self.adj, self.tadj_test, self.nsadj_test, self.features, self.labels, self.edge_index_train,\
        self.edge_train, self.edge_test, self.label_test, self.edge_index_test = load_data(args.dataset, self.repeat,args.device)
        print('Data load init finish')

    def pre_train_truck(self,epoch):
        t = time.time()
        self.truck_model.train()
        self.classify.train()
        self.optimizerTruck.zero_grad()
        self.optimizerMLP.zero_grad()
        output = self.truck_model( self.features, self.edge_index_train.cuda())
 
      
        output1 = output[self.edge_train[0].tolist()]
        output2 = output[self.edge_train[1].tolist()]
        output = torch.cat((output1,output2),1)
        output = self.classify(output)

        loss_train = self.criterionTruck(output.view(-1), self.labels.float())
        acc_train = accuracy(output, self.labels)
        loss_train.backward()
        self.optimizerTruck.step()
        self.optimizerMLP.step()
 
        loss_val = self.criterionTruck(output.view(-1), self.labels.float())
        #auc_val,fpr,tpr = auroc(output.view(-1), self.labels.float())
     
        self.truck_model.eval()
        self.classify.eval()
 
        output = self.truck_model( self.features, self.edge_index_test.cuda())
        
        output1 = output[self.edge_test[0].tolist()]
        output2 = output[self.edge_test[1].tolist()]
        output = torch.cat((output1,output2),1)
        output = self.classify(output)
   
        auc_test,fpr,tpr = auroc(output.view(-1), self.label_test.float())

        print("{ts} PRE Test set results:".format(ts='ts'),
                "auc= {:.4f}".format(auc_test.item()))

        if auc_test > self.best_teacher_truck_val:
            self.best_teacher_truck_val = auc_test
            self.teacher_truck_state = {
                'state_dict': self.truck_model.state_dict(),
                'auc_test': auc_test,
                'best_epoch': epoch+1,
                'optimizer': self.optimizerTruck.state_dict(),
            }


    def pre_train_leaf(self, epoch):

        t = time.time()
        self.sub_leafNet.train()
        self.classify.train()
        self.optimizerSub.zero_grad()
        self.optimizerMLP.zero_grad()
        output = self.sub_leafNet( self.features, self.edge_index_train.cuda())
 
      
        output1 = output[self.edge_train[0].tolist()]
        output2 = output[self.edge_train[1].tolist()]
        output = torch.cat((output1,output2),1)
        output = self.classify(output)

        loss_train = self.criterionSub(output.view(-1), self.labels.float())
        acc_train = accuracy(output, self.labels)
        loss_train.backward()
        self.optimizerSub.step()
        self.optimizerMLP.step()
 
        loss_val = self.criterionSub(output.view(-1), self.labels.float())
        #auc_val,fpr,tpr = auroc(output.view(-1), self.labels.float())
     
        self.sub_leafNet.eval()
        self.classify.eval()
 
        output = self.sub_leafNet( self.features, self.edge_index_test.cuda())

        output1 = output[self.edge_test[0].tolist()]
        output2 = output[self.edge_test[1].tolist()]
        output = torch.cat((output1,output2),1)
        output = self.classify(output)
   
        auc_test,fpr,tpr = auroc(output.view(-1), self.label_test.float())

        print("{ts} PRE Test set results:".format(ts='ts'),
                "auc= {:.4f}".format(auc_test.item()))

        if auc_test > self.best_teacher_sub_val:
            self.best_teacher_sub_val = auc_test
            self.teacher_sub_state = {
                'state_dict': self.sub_leafNet.state_dict(),
                'auc_test': auc_test,
                'best_epoch': epoch+1,
                'optimizer': self.optimizerSub.state_dict(),
            }



       
    def train_student(self, epoch):
        t = time.time()
        self.stu_model.train()
        self.truck_model.train()
        self.sub_leafNet.train()
        self.classify2.train()
        self.fuse.train()

        self.optimizerMLP.zero_grad()
        self.optimizerMLP2.zero_grad()
        self.optimizerStudent.zero_grad()
        self.optimizerTruck.zero_grad()
        self.optimizerSub.zero_grad()
        output0, middle_emb_stu = self.stu_model(self.adj, self.features)
        out1 = self.truck_model( self.features, self.edge_index_train.cuda())

        out2 = self.sub_leafNet( self.features, self.edge_index_train.cuda())
 
        emb = self.fuse(out1.unsqueeze(0).unsqueeze(0),out2.unsqueeze(0).unsqueeze(0),output0.unsqueeze(0).unsqueeze(0))
     
        output = emb
        output1 = output[self.edge_train[0].tolist()]
        output2 = output[self.edge_train[1].tolist()]
        output = torch.cat((output1,output2),1)
        output = self.classify2(output)
        middle_emb_stu = torch.cat((middle_emb_stu[1][self.edge_train[0].tolist()],middle_emb_stu[1][self.edge_train[1].tolist()]),1)

  
        contrast_fea, contrast_str = self.stu_model.loss(output0, out1, out2)
     
        loss_train = self.criterionStudent(output.view(-1), self.labels.float())
        loss_train = loss_train+contrast_fea+contrast_str
        acc_train = accuracy(output, self.labels)
        loss_train.backward()

        self.optimizerMLP.step()
        self.optimizerMLP2.step()
        self.optimizerStudent.step()
        self.optimizerTruck.step()
        self.optimizerSub.step()

    
        loss_val = self.criterionStudent(output.view(-1), self.labels.float())
        auc_val,fpr,tpr = auroc(output.view(-1), self.labels.float())
        if auc_val > self.best_student_val:
            self.best_student_val = auc_val
            self.student_state = {
                'state_dict': self.stu_model.state_dict(),
                'best_val': auc_val,
                'best_epoch': epoch+1,
                'optimizer': self.optimizerStudent.state_dict(),
            }
        print('Epoch: {:04d}'.format(epoch+1),
              'loss_train: {:.4f}'.format(loss_train.item()),
              'acc_train: {:.4f}'.format(acc_train.item()),
              'loss_val: {:.4f}'.format(loss_val.item()),
              'auc_val: {:.4f}'.format(auc_val.item()),
              'time: {:.4f}s'.format(time.time() - t))

      
        self.stu_model.eval()
        self.truck_model.eval()
        self.sub_leafNet.eval()
        self.fuse.eval()
        self.classify2.eval()

        output, _ = self.stu_model(self.nsadj_test, self.features)

        out1 = self.truck_model( self.features, self.edge_index_test.cuda())
        out2 = self.sub_leafNet( self.features, self.edge_index_test.cuda())
        emb = self.fuse(out1.unsqueeze(0).unsqueeze(0),out2.unsqueeze(0).unsqueeze(0),output.unsqueeze(0).unsqueeze(0))
        output = emb

        output1 = output[self.edge_test[0].tolist()]
        output2 = output[self.edge_test[1].tolist()]
        output = torch.cat((output1,output2),1)
        output = self.classify2(output)
      
        auc_test,fpr,tpr = auroc(output.view(-1), self.label_test.float())

        print("{ts} Test set results:".format(ts='ts'),
                "auc= {:.4f}".format(auc_test.item()))
        #self.acc_list.append(round(auc_test.item(), 4))


    def test(self, ts='teacher_fea'):
        if ts == 'teacher_fea':
            model = self.fea_model
            criterion = self.criterionTeacherFea
            model.eval()
            self.classify.eval()
            output, _ = model(self.features)
            output1 = output[self.edge_test[0].tolist()]
            output2 = output[self.edge_test[1].tolist()]
            output = torch.cat((output1,output2),1)
            output = self.classify(output)
            loss_test = criterion(output.view(-1), self.label_test.float())
            auc_test,fpr,tpr = auroc(output.view(-1), self.label_test.float())
            
            # print("{ts} Test set results:".format(ts=ts),
            #       "loss= {:.4f}".format(loss_test.item()),
            #       "auc= {:.4f}".format(auc_test.item()))
            self.acc_list_fea.append(round(auc_test.item(), 4))
        elif ts == 'teacher_str':
            model = self.str_model
            criterion = self.criterionTeacherStr
            model.eval()
            self.classify.eval()
            output,_ = model(self.tadj_test)
            output1 = output[self.edge_test[0].tolist()]
            output2 = output[self.edge_test[1].tolist()]
            output = torch.cat((output1,output2),1)  
            output = self.classify(output)          
            loss_test = criterion(output.view(-1), self.label_test.float())
            
            auc_test,fpr,tpr = auroc(output.view(-1), self.label_test.float())

            self.acc_list_str.append(round(auc_test.item(), 4))
        elif ts == 'student':
            model = self.stu_model
            criterion = self.criterionStudent
            model.eval()
            self.truck_model.eval()
            self.sub_leafNet.eval()
           
         
            self.fuse.train()
            self.classify2.eval()
            output, _ = model(self.nsadj_test, self.features)

            out1 = self.truck_model( self.features, self.edge_index_test.cuda())

            out2 = self.sub_leafNet( self.features, self.edge_index_test.cuda())
            emb = self.fuse(out1.unsqueeze(0).unsqueeze(0),out2.unsqueeze(0).unsqueeze(0),output.unsqueeze(0).unsqueeze(0))
            output = emb

            pd.DataFrame(out1.cpu().detach().numpy()).to_csv('./result/out1_truck_{}.csv'.format(str(self.repeat)),index=0,header=0)
            pd.DataFrame(out2.cpu().detach().numpy()).to_csv('./result/out2_leaf_{}.csv'.format(str(self.repeat)),index=0,header=0)
            pd.DataFrame(emb.cpu().detach().numpy()).to_csv('./result/emb_{}.csv'.format(str(self.repeat)),index=0,header=0)
            pd.DataFrame(self.edge_test[0].tolist()).to_csv('./result/index1_{}.csv'.format(str(self.repeat)),index=0,header=0)
            pd.DataFrame(self.edge_test[1].tolist()).to_csv('./result/index2_{}.csv'.format(str(self.repeat)),index=0,header=0)

            output1 = output[self.edge_test[0].tolist()]
            output2 = output[self.edge_test[1].tolist()]
            output = torch.cat((output1,output2),1)
            output = self.classify2(output)
            loss_test = criterion(output.view(-1), self.label_test.float())
            #acc_test = accuracy(output, self.labels_test)
            auc_test,fpr,tpr = auroc(output.view(-1), self.label_test.float())

            aupr,pre,re = auprc(output.view(-1), self.label_test.float())


            predlabel = output.view(-1).detach().cpu().numpy()>0.5
            predlabel = predlabel.astype(np.int32)
            labels = self.label_test.int().detach().cpu().numpy()
            acc,precision,recall,f1 = prediction(predlabel,labels)
            
            pd.DataFrame(labels).to_csv('./result/labels_{}.csv'.format(str(self.repeat)),index=0,header=0)
            pd.DataFrame(predlabel).to_csv('./result/pred_labels_{}.csv'.format(str(self.repeat)),index=0,header=0)

            print("{ts} Test set results:".format(ts=ts),
                  "loss= {:.4f}".format(loss_test.item()),
                  "auc= {:.4f}".format(auc_test.item()))
            self.acc_list.append(round(acc.item(), 4))
            self.rec_list.append(round(recall.item(), 4))
            self.prec_list.append(round(precision.item(), 4))
            self.f1_list.append(round(f1.item(), 4))
            self.auc_list.append(round(auc_test.item(), 4))
            self.aupr_list.append(round(aupr.item(), 4))
           
            self.ftr_list.append(fpr)
            self.tpr_list.append(tpr)
            self.r_list.append(re)
            self.p_list.append(pre) 

    def save_checkpoint(self, filename='./.checkpoints/'+args.dataset, ts='teacher_truck'):
        print('Save {ts} model...'.format(ts=ts))
        filename += '_{ts}'.format(ts=ts)
        if ts == 'teacher_truck':
            torch.save(self.teacher_truck_state, filename)
            print('Successfully saved feature teacher model\n...')
        elif ts == 'teacher_sub':
            torch.save(self.teacher_sub_state, filename)
            print('Successfully saved structure teacher model\n...')
        elif ts == 'student':
            torch.save(self.student_state, filename)
            print('Successfully saved student model\n...')
      
        
    def load_checkpoint(self, filename='./.checkpoints/'+ args.dataset, ts='teacher_truck'):
        print('Load {ts} model...'.format(ts=ts))
        filename += '_{ts}'.format(ts=ts)
        if ts == 'teacher_truck':
            load_state = torch.load(filename)
            self.truck_model.load_state_dict(load_state['state_dict'])
            self.optimizerTruck.load_state_dict(load_state['optimizer'])

        elif ts == 'teacher_sub':
            load_state = torch.load(filename)
            self.sub_leafNet.load_state_dict(load_state['state_dict'])
            self.optimizerSub.load_state_dict(load_state['optimizer'])

        elif ts == 'student':
            load_state = torch.load(filename)
            self.stu_model.load_state_dict(load_state['state_dict'])
            self.optimizerStudent.load_state_dict(load_state['optimizer'])

 
