
class hyperparameter:
    def __init__(self):

        self.Learning_rate = 1e-4
        self.Epoch = 200
        self.Batch_size = 64
        self.Patience = 50
        self.decay_interval = 10
        self.lr_decay = 0.5
        self.weight_decay = 1e-4
        self.embed_dim = 64
        self.protein_kernel = [4, 8, 12]
        self.drug_kernel = [4, 6, 8]
        self.conv = 40
        self.proDim = self.conv*4
        self.drugDim = self.conv*4
        self.char_dim = 64
        self.loss_epsilon = 1
        self.DEVICE = "cuda:0"
        self.drugsim_threshold = {"RDKF":0.8, "MACCS":0.7, "PMF2D":0.7}
        self.prosim_threshold ={"MF":0.75,"domain":0.9, "seq":0.99}
