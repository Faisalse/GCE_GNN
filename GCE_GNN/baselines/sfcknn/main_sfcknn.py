# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 14:18:04 2024

@author: shefai
"""
# import data preprocessing files....
from GCE_GNN.datasets.preprocessing_now_playing import *
from GCE_GNN.datasets.process_tmall_class import *
from GCE_GNN.datasets.preprocessing_digi import *
from GCE_GNN.baselines.sfcknn.sfcknn  import *
from pathlib import Path
from GCE_GNN.accuracy_measures import *

class SFCKNN_MAIN:
    
    def __init__(self, data_path, result_path, dataset = "diginetica"):
        self.dataset = dataset
        self.result_path = result_path
        if dataset == "diginetica":
            
            self.k = 1000
            self.sample_size = 2000
            self.similarity = "cosine"
        

            print("  DIGI   ")
            name = "train-item-views.csv"
            obj1 = DIGI()
            tra_sess, tes_sess, sess_clicks = obj1.data_load(data_path / name)

            tra_ids, tra_dates, all_seqs = obj1.obtian_tra(tra_sess, sess_clicks)
            tes_ids, tes_dates, tes_seqs = obj1.obtian_tes(tes_sess, sess_clicks)
            num_node = len(obj1.item_dict) + 1
            print("Number of nodes:  ", num_node)
            tr_seqs, tr_dates, tr_labs, tr_ids = obj1.process_seqs(all_seqs, tra_dates)
            self.train_data = obj1.train_convert_data_for_baselines( tr_seqs, tr_dates, tr_labs, tr_ids )
            self.unique_items_ids  = self.train_data.ItemId.unique()
            # test sequences....
            te_seqs, te_dates, te_labs, te_ids = obj1.process_seqs(tes_seqs, tes_dates)
            self.test_data = obj1.train_convert_data_for_baselines( te_seqs, te_dates, te_labs, te_ids )
            
            
            
            
        elif dataset == 'tmall':
            
            self.k = 1000
            self.sample_size = 2000
            self.similarity = "cosine"
            
            
            print("  Tamll   ")
            name = "dataset15.csv"
            obj1 = Tmall()
            tra_sess, tes_sess, sess_clicks = obj1.data_load(data_path / name)

            tra_ids, tra_dates, all_seqs = obj1.obtian_tra(tra_sess, sess_clicks)
            tes_ids, tes_dates, tes_seqs = obj1.obtian_tes(tes_sess, sess_clicks)
            num_node = len(obj1.item_dict) + 1
            tr_seqs, tr_dates, tr_labs, tr_ids = obj1.process_seqs(all_seqs, tra_dates)
            self.train_data = obj1.train_convert_data_for_baselines( tr_seqs, tr_dates, tr_labs, tr_ids )
            self.unique_items_ids  = self.train_data.ItemId.unique()
            # test sequences....
            te_seqs, te_dates, te_labs, te_ids = obj1.process_seqs(tes_seqs, tes_dates)
            
            self.test_data = obj1.train_convert_data_for_baselines( te_seqs, te_dates, te_labs, te_ids )
            
            
            
            
        elif dataset == "nowplaying":
            
            self.k = 1000
            self.sample_size = 2000
            self.similarity = "cosine"
            
            print("  Nowplaying   ")
            name = "nowplaying.csv"
            obj1 = Nowplaying()
            tra_sess, tes_sess, sess_clicks = obj1.data_load(data_path / name)


            tra_ids, tra_dates, all_seqs = obj1.obtian_tra(tra_sess, sess_clicks)
            tes_ids, tes_dates, tes_seqs = obj1.obtian_tes(tes_sess, sess_clicks)
            num_node = len(obj1.item_dict) + 1
            tr_seqs, tr_dates, tr_labs, tr_ids = obj1.process_seqs(all_seqs, tra_dates)
            
            self.train_data = obj1.train_convert_data_for_baselines( tr_seqs, tr_dates, tr_labs, tr_ids )
            
            self.unique_items_ids  = self.train_data.ItemId.unique()
        
            # test sequences....
            te_seqs, te_dates, te_labs, te_ids = obj1.process_seqs(tes_seqs, tes_dates)
            
            self.test_data = obj1.train_convert_data_for_baselines( te_seqs, te_dates, te_labs, te_ids )
            
            
        else:
            print("Mention your datatypes")
            
            
    def fit_(self, topK):
        
        obj1 = SeqFilterContextKNN(k = self.k,  sample_size = self.sample_size,  similarity = self.similarity)
        obj1.fit(self.train_data)
        session_key ='SessionId'
        time_key='Time'
        item_key= 'ItemId'
        
        # Intialize accuracy measures.....
        performance_measures = dict()
        for i in topK:
            performance_measures["MRR_"+str(i)] = MRR(i)
            performance_measures["HR_"+str(i)] = HR(i)
            
        
        test_data = self.test_data
        test_data.sort_values([session_key, time_key], inplace=True)
        items_to_predict = self.unique_items_ids
        # Previous item id and session id....
        prev_iid, prev_sid = -1, -1
        
        print("Starting predicting")
        for i in range(len(test_data)):
            sid = test_data[session_key].values[i]
            iid = test_data[item_key].values[i]
            ts = test_data[time_key].values[i]
            
            if prev_sid != sid:
                # this will be called when there is a change of session....
                prev_sid = sid
            else:
                # prediction starts from here.......
                preds = obj1.predict_next(sid, prev_iid, items_to_predict, ts)
                preds[np.isnan(preds)] = 0
    #             preds += 1e-8 * np.random.rand(len(preds)) #Breaking up ties
                preds.sort_values( ascending=False, inplace=True )    
    
                for key in performance_measures:
                    performance_measures[key].add(preds, iid)
                
                
                
                
                
            prev_iid = iid
            
        # get the results of MRR values.....
        result_frame = pd.DataFrame()    
        for key in performance_measures:
            print(key +"   "+ str(  performance_measures[key].score()    ))
            result_frame[key] =   [performance_measures[key].score()]
            
        # Intialize accuracy measures.....
        name = "GCE_GNN_SFCKNN_"+self.dataset+".txt"
        result_frame.to_csv(self.result_path /  name, sep = "\t", index = False) 
        
       
        
        
        
        
        


