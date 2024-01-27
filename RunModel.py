import os
import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from prefetch_generator import BackgroundGenerator
from sklearn.metrics import (accuracy_score, auc, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.DataProcess import get_fold_data,form_digit,\
    shuffle_dataset,CustomDataSet,CustomCollateFn,\
    CHARPROTSET,CHARISOSMISET

from utils.EarlyStoping import EarlyStopping
from utils.TestModel import test_model
from utils.ShowResult import show_result
from config import hyperparameter


def run_model(SEED, DATASET, MODEL, K_Fold):

    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    '''load hyperparameters'''
    hp = hyperparameter()
    print(f"run in {hp.DEVICE}")
    assert DATASET in ["DrugBank", "KIBA", "Enzyme", "GPCRs", "ion_channel"]
    print("Current Dataset: " + DATASET)
    """Get protein sim matrices"""
    MF_sim = pd.read_csv(f"./DataSets/{DATASET}/MF_sim.csv").to_numpy()
    domain_sim = pd.read_csv(f"./DataSets/{DATASET}/domain_sim.csv", index_col=0).to_numpy()
    seq_sim = pd.read_csv(f"./DataSets/{DATASET}/seq_sim.csv", index_col=0).to_numpy()
    np.fill_diagonal(MF_sim, 0)
    np.fill_diagonal(domain_sim, 0)
    np.fill_diagonal(seq_sim,0)
    zero = np.zeros(MF_sim.shape)
    One = np.ones(MF_sim.shape)
    MF_sim = np.where(MF_sim > hp.prosim_threshold["MF"], One, zero)
    domain_sim = np.where(domain_sim > hp.prosim_threshold["domain"], One, zero)
    seq_sim = np.where(seq_sim > hp.prosim_threshold["seq"], One, zero)

    proteinsim_dict = {"MF": MF_sim, "domain": domain_sim, "seq": seq_sim}
    """Get protein sim matrices"""
    RDKF_sim = pd.read_csv(f"./DataSets/{DATASET}/RDKF_sim.csv",index_col=0).to_numpy()
    MACCS_sim = pd.read_csv(f"./DataSets/{DATASET}/MACCS_sim.csv",index_col=0).to_numpy()
    PMF2D_sim = pd.read_csv(f"./DataSets/{DATASET}/PMF2D_sim.csv",index_col=0).to_numpy()

    np.fill_diagonal(RDKF_sim, 0)
    np.fill_diagonal(MACCS_sim, 0)
    np.fill_diagonal(PMF2D_sim, 0)
    zero = np.zeros(RDKF_sim.shape)
    One = np.ones(RDKF_sim.shape)
    RDKF_sim = np.where(RDKF_sim > hp.drugsim_threshold["RDKF"], One, zero)
    MACCS_sim = np.where(MACCS_sim > hp.drugsim_threshold["MACCS"], One, zero)
    PMF2D_sim = np.where(PMF2D_sim > hp.drugsim_threshold["PMF2D"], One, zero)

    drugsim_dict = {"RDKF":RDKF_sim, "MACCS":MACCS_sim, "PMF2D":PMF2D_sim}

    """get interaction matrix"""
    DTI = pd.read_csv(f"./DataSets/{DATASET}/interaction.csv")
    DTI = DTI.to_numpy().tolist()
    '''shuffle data'''
    DTI = shuffle_dataset(DTI, SEED)
    split_pos = len(DTI) - int(len(DTI) * 0.2)
    '''split dataset '''
    train_data_list = DTI[0:split_pos]
    test_data_list = DTI[split_pos:-1]
    print('Length of Train and Val set: {}'.format(len(train_data_list)))
    print('Length of Test set: {}'.format(len(test_data_list)))

    """convert string to integer"""
    drug_str_list = pd.read_csv(f"./DataSets/{DATASET}/drug.csv")["SMILES"].tolist()
    protein_str_list = pd.read_csv(f"./DataSets/{DATASET}/protein.csv")["sequence"].tolist()
    drug_vectors = form_digit(drug_str_list, 100, CHARISOSMISET)
    protein_vectors = form_digit(protein_str_list, 1000, CHARPROTSET)

    drug_vectors = np.array(drug_vectors)
    protein_vectors = np.array(protein_vectors)
    '''set loss function weight to deal with unbalance'''
    if DATASET == "KIBA":
        weight_loss = torch.FloatTensor([0.25, 0.75]).to(hp.DEVICE)
    else:
        weight_loss = None



    '''lists for storing metrics of the best model in each fold'''
    Accuracy_List, AUC_List, AUPR_List, Recall_List, Precision_List = [], [], [], [], []

    for i_fold in range(K_Fold):
        print('*' * 25, 'No.', i_fold + 1, '-fold', '*' * 25)

        train_dataset, valid_dataset = get_fold_data(
            i_fold, train_data_list, k=K_Fold)
        train_dataset = CustomDataSet(train_dataset)
        valid_dataset = CustomDataSet(valid_dataset)
        test_dataset = CustomDataSet(test_data_list)
        train_size = len(train_dataset)

        train_collate_fn = CustomCollateFn(drug_vectors, protein_vectors,
                                           drug_simdict=drugsim_dict,protein_simdict=proteinsim_dict,
                                           generate_batchsim=True)
        test_collate_fn = CustomCollateFn(drug_vectors, protein_vectors,generate_batchsim=False)
        train_dataset_loader = DataLoader(train_dataset, batch_size=hp.Batch_size, shuffle=True, num_workers=0,
                                          collate_fn=train_collate_fn, drop_last=True)
        valid_dataset_loader = DataLoader(valid_dataset, batch_size=hp.Batch_size, shuffle=False, num_workers=0,
                                          collate_fn=test_collate_fn, drop_last=True)
        test_dataset_loader = DataLoader(test_dataset, batch_size=hp.Batch_size, shuffle=False, num_workers=0,
                                         collate_fn=test_collate_fn, drop_last=True)
        train_dataset_loader_for_test =  DataLoader(train_dataset, batch_size=hp.Batch_size, shuffle=False, num_workers=0,
                                          collate_fn=test_collate_fn, drop_last=True)

        """ create model"""
        model = MODEL(hp).to(hp.DEVICE)

        """Initialize weights"""
        weight_p, bias_p = [], []
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for name, p in model.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]


        optimizer = optim.AdamW(
            [{'params': weight_p, 'weight_decay': hp.weight_decay}, {'params': bias_p, 'weight_decay': 0}], lr=hp.Learning_rate)

        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=hp.Learning_rate, max_lr=hp.Learning_rate*10, cycle_momentum=False,
                                                step_size_up=train_size // hp.Batch_size)

        Loss = nn.CrossEntropyLoss(weight=weight_loss)

        """Output files"""
        save_path = "./output/" + DATASET + "/{}".format(i_fold+1)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        file_results = save_path + '/' + 'The_results_of_whole_dataset.txt'

        early_stopping = EarlyStopping(
            savepath=save_path, patience=hp.Patience, verbose=True, delta=0)

        """Start training."""
        print('Training...')
        for epoch in range(1, hp.Epoch + 1):
            if early_stopping.early_stop == True:
                break
            train_pbar = tqdm(
                enumerate(BackgroundGenerator(train_dataset_loader)),
                total=len(train_dataset_loader))

            """train"""
            train_losses_in_epoch = []
            model.train()
            for train_i, train_data in train_pbar:
                train_drugs, train_proteins, train_labels,batch_drugsim_dict,batch_proteinsim_dict = train_data
                train_drugs = train_drugs.to(hp.DEVICE)
                train_proteins = train_proteins.to(hp.DEVICE)
                train_labels = train_labels.to(hp.DEVICE)
                for key in batch_drugsim_dict.keys():
                    batch_drugsim_dict[key] = batch_drugsim_dict[key].to(hp.DEVICE)
                for key in batch_proteinsim_dict.keys():
                    batch_proteinsim_dict[key] = batch_proteinsim_dict[key].to(hp.DEVICE)
                optimizer.zero_grad()
                predicted_interaction, ContrastiveLoss = model(train_drugs, train_proteins, batch_drugsim_dict, batch_proteinsim_dict,contrast_loss=True)
                train_loss = Loss(predicted_interaction, train_labels) + ContrastiveLoss
                train_losses_in_epoch.append(train_loss.item())
                train_loss.backward()
                # torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10, norm_type=2)
                optimizer.step()
                scheduler.step()
            train_loss_a_epoch = np.average(
                train_losses_in_epoch)  # 一次epoch的平均训练loss

            """valid"""
            valid_pbar = tqdm(
                enumerate(BackgroundGenerator(valid_dataset_loader)),
                total=len(valid_dataset_loader))
            valid_losses_in_epoch = []
            model.eval()
            Y, P, S = [], [], []
            with torch.no_grad():
                for valid_i, valid_data in valid_pbar:

                    valid_drugs, valid_proteins, valid_labels = valid_data

                    valid_drugs = valid_drugs.to(hp.DEVICE)
                    valid_proteins = valid_proteins.to(hp.DEVICE)
                    valid_labels = valid_labels.to(hp.DEVICE)

                    valid_scores = model(valid_drugs, valid_proteins,contrast_loss=False)
                    valid_loss = Loss(valid_scores, valid_labels)
                    valid_losses_in_epoch.append(valid_loss.item())
                    valid_labels = valid_labels.to('cpu').data.numpy()
                    valid_scores = F.softmax(
                        valid_scores, 1).to('cpu').data.numpy()
                    valid_predictions = np.argmax(valid_scores, axis=1)
                    valid_scores = valid_scores[:, 1]

                    Y.extend(valid_labels)
                    P.extend(valid_predictions)
                    S.extend(valid_scores)

            Precision_dev = precision_score(Y, P)
            Reacll_dev = recall_score(Y, P)
            Accuracy_dev = accuracy_score(Y, P)

            AUC_dev = roc_auc_score(Y, S)

            tpr, fpr, _ = precision_recall_curve(Y, S)
            PRC_dev = auc(fpr, tpr)
            valid_loss_a_epoch = np.average(valid_losses_in_epoch)

            epoch_len = len(str(hp.Epoch))
            print_msg = (f'[{epoch:>{epoch_len}}/{hp.Epoch:>{epoch_len}}] ' +
                         f'train_loss: {train_loss_a_epoch:.5f} ' +
                         f'valid_loss: {valid_loss_a_epoch:.5f} ' +
                         f'valid_AUC: {AUC_dev:.5f} ' +
                         f'valid_PRC: {PRC_dev:.5f} ' +
                         f'valid_Accuracy: {Accuracy_dev:.5f} ' +
                         f'valid_Precision: {Precision_dev:.5f} ' +
                         f'valid_Reacll: {Reacll_dev:.5f} ')
            print(print_msg)

            '''save checkpoint and make decision when early stop'''
            early_stopping(Accuracy_dev, model, epoch)

        '''load best checkpoint'''
        model.load_state_dict(torch.load(
            early_stopping.savepath + '/valid_best_checkpoint.pth'))

        '''test model'''
        trainset_test_results, _, _, _, _, _ = test_model(
            model, train_dataset_loader_for_test, save_path, DATASET, Loss, hp.DEVICE, dataset_class="Train")
        validset_test_results, _, _, _, _, _ = test_model(
            model, valid_dataset_loader, save_path, DATASET, Loss, hp.DEVICE, dataset_class="Valid")
        testset_test_results, Accuracy_test, Precision_test, Recall_test, AUC_test, PRC_test = test_model(
            model, test_dataset_loader, save_path, DATASET, Loss, hp.DEVICE, dataset_class="Test")
        AUC_List.append(AUC_test)
        Accuracy_List.append(Accuracy_test)
        AUPR_List.append(PRC_test)
        Recall_List.append(Recall_test)
        Precision_List.append(Precision_test)
        with open(file_results, 'a') as f:
            f.write("test results in each dataset" + '\n')
            f.write(trainset_test_results + '\n')
            f.write(validset_test_results + '\n')
            f.write(testset_test_results + '\n')

    show_result(DATASET, Accuracy_List, Precision_List,
                Recall_List, AUC_List, AUPR_List)

if __name__ =="__main__":
    from model import CLSF_DTI
    for dataset in ["DrugBank", "Enzyme", "GPCRs", "ion_channel", "KIBA"]:
        run_model(114514,dataset,CLSF_DTI,5)
