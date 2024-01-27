import torch
import torch.nn as nn


class DrugFeature(nn.Module):

    def __init__(self, hp, ):
        super().__init__()
        self.dim = hp.char_dim
        self.conv = hp.conv
        self.drug_kernel = hp.drug_kernel
        self.drug_vocab_size = 65
        self.drug_embed = nn.Embedding(
            self.drug_vocab_size, self.dim, padding_idx=0)
        self.Drug_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv,
                      kernel_size=self.drug_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2,
                      kernel_size=self.drug_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv * 4,
                      kernel_size=self.drug_kernel[2]),
            nn.ReLU(),
        )

    def forward(self, drug):

        drugembed = self.drug_embed(drug)
        drugembed = drugembed.permute(0, 2, 1)
        drugConv = self.Drug_CNNs(drugembed)

        return drugConv


class ProteinFeature(nn.Module):
    def __init__(self, hp, ):
        super().__init__()
        self.dim = hp.char_dim
        self.conv = hp.conv
        self.protein_vocab_size = 26
        self.protein_kernel = hp.protein_kernel
        self.protein_embed = nn.Embedding(
            self.protein_vocab_size, self.dim, padding_idx=0)
        self.Protein_CNNs = nn.Sequential(
            nn.Conv1d(in_channels=self.dim, out_channels=self.conv,
                      kernel_size=self.protein_kernel[0]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv, out_channels=self.conv * 2,
                      kernel_size=self.protein_kernel[1]),
            nn.ReLU(),
            nn.Conv1d(in_channels=self.conv * 2, out_channels=self.conv * 4,
                      kernel_size=self.protein_kernel[2]),
            nn.ReLU(),
        )

    def forward(self, protein):

        proteinembed = self.protein_embed(protein)
        proteinembed = proteinembed.permute(0, 2, 1)
        proteinConv = self.Protein_CNNs(proteinembed)
        return proteinConv


class CrossAttention(nn.Module):

    def __init__(self, hp,  protein_MAX_LENGH=1000,
                 drug_MAX_LENGH=100):
        super().__init__()
        self.protein_MAX_LENGTH = protein_MAX_LENGH
        self.drug_MAX_LENGTH = drug_MAX_LENGH
        self.conv = hp.conv
        self.attention_dim = hp.conv * 4
        self.mix_attention_head = 5
        self.drug_kernel = hp.drug_kernel
        self.protein_kernel = hp.protein_kernel
        self.mix_attention_layer = nn.MultiheadAttention(
            self.attention_dim, self.mix_attention_head)
        self.drug_dim_afterCNNs = self.drug_MAX_LENGTH - \
                                  self.drug_kernel[0] - self.drug_kernel[1] - self.drug_kernel[2] + 3
        self.protein_dim_afterCNNs = self.protein_MAX_LENGTH - \
                                     self.protein_kernel[0] - self.protein_kernel[1] - \
                                     self.protein_kernel[2] + 3
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.leaky_relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(self.conv * 8, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 2)

        self.Drug_max_pool = nn.MaxPool1d(self.drug_dim_afterCNNs)
        self.Protein_max_pool = nn.MaxPool1d(self.protein_dim_afterCNNs)

    def forward(self, drugConv, proteinConv):

        drug_QKV = drugConv.permute(2, 0, 1)
        protein_QKV = proteinConv.permute(2, 0, 1)
        drug_att, _ = self.mix_attention_layer(drug_QKV, protein_QKV, protein_QKV)
        protein_att, _ = self.mix_attention_layer(protein_QKV, drug_QKV, drug_QKV)
        drug_att = drug_att.permute(1, 2, 0)
        protein_att = protein_att.permute(1, 2, 0)
        drugConv = drugConv * 0.5 + drug_att * 0.5
        proteinConv = proteinConv * 0.5 + protein_att * 0.5
        drugConv = self.Drug_max_pool(drugConv).squeeze(2)
        proteinConv = self.Protein_max_pool(proteinConv).squeeze(2)

        pair = torch.cat([drugConv, proteinConv], dim=1)
        pair = self.dropout1(pair)
        fully1 = self.leaky_relu(self.fc1(pair))
        fully1 = self.dropout2(fully1)
        fully2 = self.leaky_relu(self.fc2(fully1))
        fully2 = self.dropout3(fully2)
        fully3 = self.leaky_relu(self.fc3(fully2))
        predict = self.out(fully3)

        return predict, drugConv, proteinConv


class BatchContrast(nn.Module):
    """
    提取相似性网络中的蛋白质特征，
    并计算对比损失
    """
    def __init__(self, hp,  ProKeys, DrugKeys,
                 tau=0.07):
        """
        :param batch_size:  节点特征的处理批量大小
        :param Keys:  相似性网络的名字
        :param tau:  蒸馏温度
        """
        super().__init__()
        self.hp = hp
        self.tau = tau

        self.ProMLP = nn.ModuleDict({k: nn.Sequential(
            nn.Linear(hp.proDim, hp.proDim*2),
            nn.LeakyReLU(),
            nn.Linear(hp.proDim*2,hp.proDim*4),)for k in ProKeys})

        self.DrugMLP = nn.ModuleDict({k: nn.Sequential(
            nn.Linear(hp.drugDim, hp.drugDim * 2),
            nn.LeakyReLU(),
            nn.Linear(hp.drugDim * 2, hp.drugDim * 4), ) for k in DrugKeys})


    def sim(self, z1, z2):

        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def compute_loss(self, Feature_Matrix, pos_matrix):

        sim_matrix = self.sim(Feature_Matrix, Feature_Matrix)

        sim_matrix= sim_matrix / (torch.sum(sim_matrix.reshape(1, -1), dim=-1) + 1e-8)

        loss= -torch.log(sim_matrix.mul(pos_matrix).sum(dim=-1) + 1e-8).mean()

        return loss

    def forward(self, ProF, DrugF, pro_sim_dict, drug_sim_dict):

        loss = 0
        ProfDict = {}
        DrugFdict = {}
        for key in pro_sim_dict.keys():
            ProfDict[key] = self.ProMLP[key](ProF)

        for key in drug_sim_dict.keys():
            DrugFdict[key] = self.DrugMLP[key](DrugF)

        for key in pro_sim_dict.keys():
                loss = loss + self.compute_loss(ProfDict[key],pro_sim_dict[key])

        for key in drug_sim_dict.keys():
                loss = loss + self.compute_loss(DrugFdict[key], drug_sim_dict[key])

        return loss


class CLSF_DTI(nn.Module):

    def __init__(self, hp, tau=0.07):
        super().__init__()
        self.Drug_extract = DrugFeature(hp)
        self.Protein_extract = ProteinFeature(hp)
        self.ContrastLayer = BatchContrast(hp=hp,ProKeys=hp.prosim_threshold.keys(),DrugKeys=hp.drugsim_threshold.keys(), tau=tau)
        self.fusion = CrossAttention(hp)


    def forward(self, Drug_x, Protein_batch_x, drug_sim_dict=None, pro_sim_dict=None,  contrast_loss=True):
        DrugF = self.Drug_extract(Drug_x)
        ProteinF = self.Protein_extract(Protein_batch_x)

        predict, DrugF, ProteinF = self.fusion(DrugF, ProteinF)
        if contrast_loss:
            ContrastiveLoss = self.ContrastLayer(ProF=ProteinF, DrugF=DrugF,
                                                 pro_sim_dict=pro_sim_dict, drug_sim_dict=drug_sim_dict)
            return predict, ContrastiveLoss
        else:
            return predict



