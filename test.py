import os
import pandas as pd
from torch.autograd import Variable
from sklearn import metrics
from MEGPPIS_model import *
import pickle
import csv
import numpy as np
import matplotlib.pyplot as plt
import time
# Path
Dataset_Path = "./Dataset/"

Model_Path = "./Log/test_model/model/"

print(Model_Path)

time_data = [0,0,0,0]


def save_time_data(time_list,seq_lenth,datapath):
    time_data = {}
    time_data['time_list'] = time_list
    time_data['seq_lenth'] = seq_lenth
    with open(datapath, "wb") as f:
        pickle.dump(time_data, f)



def evaluate(model, data_loader):
    model.eval()

    epoch_loss = 0.0
    n = 0
    valid_pred = []
    valid_true = []
    pred_dict = {}
    test_dic = {}
    agat_dic = {}
    true_dic = {}
    time_list = []
    seq_lenth = []
    
    for data in data_loader:
        with torch.no_grad():
            sequence_names, seq, labels, node_features, G_batch, adj_matrix, pos = data
           
            if torch.cuda.is_available():
                node_features = Variable(node_features.cuda().float())
                adj_matrix = Variable(adj_matrix.cuda())
                G_batch.edata['ex'] = Variable(G_batch.edata['ex'].float())
                G_batch = G_batch.to(torch.device('cuda:0'))
                y_true = Variable(labels.cuda())
                pos =Variable(pos.cuda().float())


            else:
                node_features = Variable(node_features.float())
                adj_matrix = Variable(adj_matrix)
                y_true = Variable(labels)
                G_batch.edata['ex'] = Variable(G_batch.edata['ex'].float())

            adj_matrix = torch.squeeze(adj_matrix)
            y_true = torch.squeeze(y_true)
            y_true = y_true.long()

            y_pred = model(node_features, G_batch, adj_matrix, pos)
            
            
            loss = model.criterion(y_pred, y_true)
            softmax = torch.nn.Softmax(dim=1)
            y_pred = softmax(y_pred)
            y_pred = y_pred.cpu().detach().numpy()
            y_true = y_true.cpu().detach().numpy()
            valid_pred += [pred[1] for pred in y_pred]
            valid_true += list(y_true)
            pred_dict[sequence_names[0]] = [pred[1] for pred in y_pred]


            test_pred = [pred[1] for pred in y_pred]
            test_dic[sequence_names[0]] = [1 if pr >= 0.18  else 0 for pr in test_pred]
            true_dic[sequence_names[0]] = list(y_true)

            epoch_loss += loss.item()

            n += 1
    

    epoch_loss_avg = epoch_loss / n

    f_agat = "./Sites/agat.p"

    # exit()
    with open(f_agat, "wb") as f:
        pickle.dump(test_dic, f)
    
    
    return epoch_loss_avg, valid_true, valid_pred, pred_dict


def plot(y_test,y_score):

    fpr, tpr, thread = metrics.roc_curve(y_test, y_score)
    roc_auc = metrics.auc(fpr, tpr)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig('AUC.png',)

def save_plot_data(binary_true, y_pred, datapath):

    dic = {}
    dic['y_true'] = binary_true
    dic['y_pred'] = y_pred
    with open(datapath, "wb") as f:
        pickle.dump(dic, f)



def analysis(y_true, y_pred, best_threshold = None):
    if best_threshold == None:
        best_f1 = 0
        best_threshold = 0
        for threshold in range(0, 100):
            threshold = threshold / 100
            binary_pred = [1 if pred >= threshold else 0 for pred in y_pred]
            binary_true = y_true
            f1 = metrics.f1_score(binary_true, binary_pred)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold


    
    binary_pred = [1 if pred >= best_threshold else 0 for pred in y_pred]
    binary_true = y_true
    
    # binary evaluate
    binary_acc = metrics.accuracy_score(binary_true, binary_pred)
    precision = metrics.precision_score(binary_true, binary_pred)
    recall = metrics.recall_score(binary_true, binary_pred)
    f1 = metrics.f1_score(binary_true, binary_pred)
    AUC = metrics.roc_auc_score(binary_true, y_pred)
    precisions, recalls, thresholds = metrics.precision_recall_curve(binary_true, y_pred)
    AUPRC = metrics.auc(recalls, precisions)
    mcc = metrics.matthews_corrcoef(binary_true, binary_pred)

    results = {
        'binary_acc': binary_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'AUC': AUC,
        'AUPRC': AUPRC,
        'mcc': mcc,
        'threshold': best_threshold
    }
    return results

    

def test(test_dataframe, psepos_path, dataset_name):
    test_loader = DataLoader(dataset=ProDataset(dataframe=test_dataframe,psepos_path=psepos_path), batch_size=BATCH_SIZE, shuffle=True, num_workers=2, collate_fn=graph_collate)

    for model_name in sorted(os.listdir(Model_Path)):
        print(model_name)
        
        model = MEGPPIS(LAYER, INPUT_DIM, HIDDEN_DIM, NUM_CLASSES, DROPOUT, LAMBDA, ALPHA)
        if torch.cuda.is_available():
            model.cuda()
        model.load_state_dict(torch.load(Model_Path + model_name, map_location='cuda:0'))
        start_time = time.time()  
        epoch_loss_test_avg, test_true, test_pred, pred_dict = evaluate(model, test_loader)
        end_time = time.time()  
        elapsed_time = end_time - start_time  
        print(elapsed_time)

        if dataset_name == 'Test_60':
            time_data[0] += elapsed_time
        elif dataset_name == 'Test_315-28':
            time_data[1] += elapsed_time
        elif dataset_name == 'Btest_31-6':
            time_data[2] += elapsed_time
        else:
            time_data[3] += elapsed_time
        # exit()

        result_test = analysis(test_true, test_pred)
 
        
        print("========== Evaluate Test set ==========")
        print("Test loss: ", epoch_loss_test_avg)
        print("Test binary acc: ", result_test['binary_acc'])
        print("Test precision:", result_test['precision'])
        print("Test recall: ", result_test['recall'])
        print("Test f1: ", result_test['f1'])
        print("Test AUC: ", result_test['AUC'])
        print("Test AUPRC: ", result_test['AUPRC'])
        print("Test mcc: ", result_test['mcc'])
        print("Threshold: ", result_test['threshold'])
        

def test_one_dataset(dataset, psepos_path, dataset_name):
    IDs, sequences, labels = [], [], []
    for ID in dataset:
        IDs.append(ID)
        item = dataset[ID]
        sequences.append(item[0])
        labels.append(item[1])
    test_dic = {"ID": IDs, "sequence": sequences, "label": labels}
    test_dataframe = pd.DataFrame(test_dic)
    test(test_dataframe, psepos_path, dataset_name)


def main():
    with open(Dataset_Path + "Test_60.pkl", "rb") as f:
        Test_60 = pickle.load(f)

    with open(Dataset_Path + "Test_315-28.pkl", "rb") as f:
        Test_315_28 = pickle.load(f)

    with open(Dataset_Path + "UBtest_31-6.pkl", "rb") as f:
        UBtest_31_6 = pickle.load(f)

    Btest_31_6 = {}
    with open(Dataset_Path + "bound_unbound_mapping31-6.txt", "r") as f:
        lines = f.readlines()[1:]
    for line in lines:
        bound_ID, unbound_ID, _ = line.strip().split()
        Btest_31_6[bound_ID] = Test_60[bound_ID]

    Test60_psepos_Path = './Feature/psepos/Test60_psepos_SC.pkl'
    Test315_28_psepos_Path = './Feature/psepos/Test315-28_psepos_SC.pkl'
    Btest31_psepos_Path = './Feature/psepos/Test60_psepos_SC.pkl'
    UBtest31_28_psepos_Path = './Feature/psepos/UBtest31-6_psepos_SC.pkl'



    print("Evaluate GraphPPIS on Test_60")
    test_one_dataset(Test_60, Test60_psepos_Path, 'Test_60')
    exit()
    # print("Evaluate GraphPPIS on Test_315-28")
    # test_one_dataset(Test_315_28, Test315_28_psepos_Path,'Test_315-28')

    # print("Evaluate GraphPPIS on Btest_31-6")
    # test_one_dataset(Btest_31_6, Btest31_psepos_Path, 'Btest_31-6')

    # print("Evaluate GraphPPIS on UBtest_31-6")
    # test_one_dataset(UBtest_31_6, UBtest31_28_psepos_Path, 'UBtest_31-6')






if __name__ == "__main__":
    main()
