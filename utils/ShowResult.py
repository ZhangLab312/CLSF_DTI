import numpy as np


def show_result(DATASET, Accuracy_List, Precision_List, Recall_List, AUC_List, AUPR_List,exp_name):
    Accuracy_mean, Accuracy_std = np.mean(Accuracy_List), np.std(Accuracy_List)
    Precision_mean, Precision_std = np.mean(
        Precision_List), np.std(Precision_List)
    Recall_mean, Recall_std = np.mean(Recall_List), np.std(Recall_List)
    AUC_mean, AUC_std = np.mean(AUC_List), np.std(AUC_List)
    PRC_mean, PRC_std = np.mean(AUPR_List), np.std(AUPR_List)


    print("The model's results:")
    filepath = "./output/{}/results.txt".format(DATASET)

    with open(filepath, 'a') as f:
        f.write(f"exp{exp_name}:\n")
        f.write('Accuracy(std):{:.4f}({:.4f})'.format(
            Accuracy_mean, Accuracy_std) + '\n')
        f.write('Precision(std):{:.4f}({:.4f})'.format(
            Precision_mean, Precision_std) + '\n')
        f.write('Recall(std):{:.4f}({:.4f})'.format(
            Recall_mean, Recall_std) + '\n')
        f.write('AUC(std):{:.4f}({:.4f})'.format(AUC_mean, AUC_std) + '\n')
        f.write('PRC(std):{:.4f}({:.4f})'.format(PRC_mean, PRC_std) + '\n')
    print('Accuracy(std):{:.4f}({:.4f})'.format(Accuracy_mean, Accuracy_std))
    print('Precision(std):{:.4f}({:.4f})'.format(
        Precision_mean, Precision_std))
    print('Recall(std):{:.4f}({:.4f})'.format(Recall_mean, Recall_std))
    print('AUC(std):{:.4f}({:.4f})'.format(AUC_mean, AUC_std))
    print('PRC(std):{:.4f}({:.4f})'.format(PRC_mean, PRC_std))


