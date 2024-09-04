import time
import argparse
import pickle
from GCE_GNN.model import *
from GCE_GNN.utils import *
from pathlib import Path
import torch
# import data preprocessing files....
from GCE_GNN.datasets.preprocessing_now_playing import *
from GCE_GNN.datasets.process_tmall_class import *
from GCE_GNN.datasets.preprocessing_digi import *
# sfcknn model
from GCE_GNN.baselines.sfcknn.main_sfcknn import *
# baseline models
from GCE_GNN.baselines.SR.main_sr import *
# vstan model
from GCE_GNN.baselines.vstan.main_vstan import *
#stan model....
from GCE_GNN.baselines.stan.main_stan import *
# import accuracy measures
from GCE_GNN.accuracy_measures import *
def init_seed(seed=None):
    if seed is None:
        seed = int(time.time() * 1000 // 1000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica', help='diginetica/nowplaying/tmall')
parser.add_argument('--hiddenSize', type=int, default=100)
parser.add_argument('--epoch', type=int, default=50)
parser.add_argument('--activate', type=str, default='relu')
parser.add_argument('--n_sample_all', type=int, default=12)
parser.add_argument('--n_sample', type=int, default=12)
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.001, help='learning rate.')
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay.')
parser.add_argument('--lr_dc_step', type=int, default=3, help='the number of steps after which the learning rate decay.')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty ')
parser.add_argument('--n_iter', type=int, default=2)
parser.add_argument('--dropout_gcn', type=float, default=0, help='Dropout rate.')      
parser.add_argument('--dropout_local', type=float, default=0, help='Dropout rate.')     
parser.add_argument('--dropout_global', type=float, default=0.6, help='Dropout rate.')
parser.add_argument('--validation', action='store_true', help='validation')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=3)
parser.add_argument('--MRR', type=float, default=[5, 10, 20], help='learning rate')  
parser.add_argument('--HR', type=float, default=[5, 10, 20], help='learning rate') 
parser.add_argument('--topkList', type=float, default=[5, 10, 20], help='learning rate') 
opt = parser.parse_args()
data_path = Path("data/")
data_path = data_path.resolve()
result_path = Path("results/")
result_path = result_path.resolve()
def run_experiments_for_GCE_GNN():
    print(opt)

    init_seed(2020) 
    if opt.dataset == 'diginetica':
        name = "train-item-views.csv"
        obj1 = DIGI()
        tra_sess, tes_sess, sess_clicks = obj1.data_load(data_path / name)
        tra_ids, tra_dates, all_seqs = obj1.obtian_tra(tra_sess, sess_clicks)
        tes_ids, tes_dates, tes_seqs = obj1.obtian_tes(tes_sess, sess_clicks)
        num_node = len(obj1.item_dict) + 1
        print("Number of nodes:  ", num_node)
        tr_seqs, tr_dates, tr_labs, tr_ids = obj1.process_seqs(all_seqs, tra_dates)
        # test sequences....
        te_seqs, te_dates, te_labs, te_ids = obj1.process_seqs(tes_seqs, tes_dates)
        opt.n_iter = 2
        opt.dropout_gcn = 0.2
        opt.dropout_local = 0.0
        
    elif opt.dataset == 'nowplaying':
        # dataset path....
        name = "nowplaying.csv"
        obj1 = Nowplaying()
        tra_sess, tes_sess, sess_clicks = obj1.data_load(data_path / name)
        tra_ids, tra_dates, all_seqs = obj1.obtian_tra(tra_sess, sess_clicks)
        tes_ids, tes_dates, tes_seqs = obj1.obtian_tes(tes_sess, sess_clicks)
        num_node = len(obj1.item_dict) + 1
        tr_seqs, tr_dates, tr_labs, tr_ids = obj1.process_seqs(all_seqs, tra_dates)
        # test sequences....
        te_seqs, te_dates, te_labs, te_ids = obj1.process_seqs(tes_seqs, tes_dates)
        opt.n_iter = 1
        opt.dropout_gcn = 0.0
        opt.dropout_local = 0.0
            
    elif opt.dataset == 'tmall':
        # dataset

        name = "dataset15.csv"
        obj1 = Tmall()
        tra_sess, tes_sess, sess_clicks = obj1.data_load(data_path / name)
        tra_ids, tra_dates, all_seqs = obj1.obtian_tra(tra_sess, sess_clicks)
        tes_ids, tes_dates, tes_seqs = obj1.obtian_tes(tes_sess, sess_clicks)
        num_node = len(obj1.item_dict) + 1
        tr_seqs, tr_dates, tr_labs, tr_ids = obj1.process_seqs(all_seqs, tra_dates)
        # test sequences....
        te_seqs, te_dates, te_labs, te_ids = obj1.process_seqs(tes_seqs, tes_dates)
        opt.n_iter = 1
        opt.dropout_gcn = 0.6
        opt.dropout_local = 0.5
        
    else:
        num_node = 310
    
    adj, num = build_graph(num_node, all_seqs)
    train_data = (tr_seqs, tr_labs)
    test_data = (te_seqs, te_labs)
    train_tr, train_val = split_validation(train_data, opt.valid_portion)
    train_tr = Data(train_tr)
    train_val = Data(train_val)
    train_data = Data(train_data)
    test_data = Data(test_data)
    adj, num = handle_adj(adj, num_node, opt.n_sample_all, num)
    model = trans_to_cuda(CombineGraph(opt, num_node, adj, num))

    early_stopping = EarlyStopping() 
    print('Start traing on validation data to best epoch value through early stopping')
    for epoch in range(opt.epoch):
        model.train()
        model = model_training(model, train_tr)
        peroformance_measure = testing(model, train_val)
        early_stopping(peroformance_measure['HR_20'].score(), epoch)
        if early_stopping.early_stop:
            break
    print("train model on best epoch value.....")
    for epoch in range(early_stopping.epoch+1):
        model = model_training(model, train_data)
    peroformance_measure = testing(model, test_data)
    result_frame = pd.DataFrame()    
    for key in peroformance_measure:
        print(key +"   "+ str(  peroformance_measure[key].score()))
        result_frame[key] =   [peroformance_measure[key].score()]
    name = "GCE_GNN_"+opt.dataset+".txt"
    print(result_frame)
    result_frame.to_csv(result_path/name, sep='\t', index = False) 
    
def testing(model, test_data):
    peroformance_measure = dict()
    for i in opt.topkList:
        peroformance_measure["MRR_"+str(i)] = MRR(i)
        peroformance_measure["HR_"+str(i)] = HR(i)
    model.eval()
    test_loader = torch.utils.data.DataLoader(test_data, num_workers=4, batch_size=1,shuffle=False, pin_memory=True)                                
    for data in test_loader:

        target, scores = forward(model, data)
        sub_scores = scores.topk(40)[1]
        sub_scores = trans_to_cpu(sub_scores).detach().numpy()
        sub_scores = sub_scores.flatten()
        target = target.numpy()
        target = target[0] - 1
        recommendation_list = pd.Series([0 for i in range(len(sub_scores))], index = sub_scores)
        for key in peroformance_measure:
            peroformance_measure[key].add(recommendation_list, target)
    return peroformance_measure
if __name__ == '__main__':
    
    print("Experiments are runing for each model. After execution, the results will be saved into *results*. Thanks for patience.")
    print("Experiments are runinig for GCE_GNN model................... wait for results...............")
    run_experiments_for_GCE_GNN()
    print("Experiments are runinig for SR model................... wait for results...............")
    se_obj = SequentialRulesMain(data_path, result_path, dataset = opt.dataset)
    se_obj.fit_(opt.topkList)
    print("Experiments are runinig for VSTAN model................... wait for results...............")
    vstan_obj = VSTAN_MAIN(data_path, result_path, dataset = opt.dataset)
    vstan_obj.fit_(opt.topkList)
    print("Experiments are runinig for STAN model................... wait for results...............")
    stan_obj = STAN_MAIN(data_path, result_path, dataset = opt.dataset)
    stan_obj.fit_(opt.topkList)
    print("Experiments are runinig for SFCKNN model................... wait for results...............")
    sfcknn_obj = SFCKNN_MAIN(data_path, result_path, dataset = opt.dataset)
    sfcknn_obj.fit_(opt.topkList)
    
