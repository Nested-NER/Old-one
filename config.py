
import os

vec_file = {"wiki-pubmed": "./model/word2vec/wikipedia-pubmed-and-PMC-w2v.bin",
            "glove": "./model/word2vec/glove_word2vec_100d.txt"
            }


class Config:
    def __init__(self):
        # sample
        self.data_set = "GENIA"  # ACE05 ACE04 GENIA conll2003
        self.data_path = f"./dataset/{self.data_set}/"
        self.Lb = 10  # 10 8 6
        self.train_pos_iou_th = 1
        self.train_neg_iou_th = 0.86  if self.data_set == "GENIA"  else 0.81 # 0.01 0.51 0.67 0.76 0.81 0.86 0.91 0.99     0.804-unk-genia
        self.mode = None

        # embedding
        # bert
        self.use_bert = False
        self.bert_config = 'large'  # "base" "large"
        self.fusion = True
        self.fusion_sum = True
        self.use_last_four = False
        self.input_size_bert = 768 if self.bert_config == 'base' else 1024
        self.fusion_layer = 13 if self.bert_config == 'base' else 25
        # word vector
        self.vec_model = "wiki-pubmed" if self.data_set == "GENIA"  else "glove" # glove_200d  glove wiki PubMed
        self.word_embedding_size = 100 if self.vec_model == "glove" else 200
        self.word2vec_path = vec_file[self.vec_model]


        # model
        self.use_cnn = True
        self.cnn_block = 5 # 4
        self.kernel_size = 3  # 3
        self.layer2_pooling =  3 # 1

        # DTE
        self.if_DTE = True
        self.if_char = True
        self.char_embedding_size = 25 #25  + 4*5
        self.if_pos = True
        self.pos_embedding_size = 6 #6 +4*3
        self.if_transformer = True
        self.N = 2
        self.h = 4
        self.if_bidirectional = True

        #multitask
        self.MultiTaskLearning = False  # True
        self.nested_depth = 3
        self.nested_depth_fc_size = 1024  if self.use_bert == True else 256 #  256
        self.weight_layer = 0.2

        # train
        self.if_gpu = True
        self.if_shuffle = True
        self.if_freeze = False if self.vec_model == "glove" else True
        self.dropout = 0.5
        self.epoch = 100 if self.vec_model == "glove" else 100 #100
        self.batch_size = 8 # 12
        self.opt ="Adam" #
        self.lr = 3e-4  if self.use_bert == False or self.vec_model == "glove" else  3e-4 # 0.005 1e-4
        self.score_th = 0.91 if self.vec_model == "glove" else 0.65

        # test
        self.if_output = False
        self.if_detail = True
        self.if_filter = True
        self.if_filter_single_layer = False
        self.layer_minlen = [1, 2, 3]
        self.layer_maxlen = [self.Lb, self.Lb, self.Lb]
        self.test_model_path = "./model/" + self.data_set + '/' + 'epoch80_f1_0.771.pth'
        self.softmax_threshold = 0  #0.5

    def __repr__(self):
        return str(vars(self))

    def get_pkl_path(self, mode):
        path = self.data_path
        if mode == "word2vec":
            path += f"word_vec"
        else:
            if mode == "config":
                path += f"config"
            else:
                path += mode + "/" \
                        + f"C_:{self.Lb}" \
                        + f"_multilayer:{self.MultiTaskLearning}"
                if mode == "train":
                    path += f"_neg_iou_th{self.train_neg_iou_th}"

        return path + f"_{self.vec_model}.pkl"

    def get_model_path(self):
        path = f"./model/{self.data_set}/"
        path += f"{self.vec_model}/" \
                + f"pooling_size:{self.layer2_pooling}_" \
                + f"kernel_size:{self.kernel_size}_" \
                + f"cnn_block:{self.cnn_block}_" \
                  f"weight_layer:{self.weight_layer}_" \
                  f"C:{self.Lb}_" \
                  f"IOU:{self.train_neg_iou_th}_" \
                + f"multi_layer:{self.MultiTaskLearning}_" \
                + f"use_fusion:{self.fusion}_" \
                + f"use_cnn:{self.use_cnn}_" \
                + f"use_bert:{self.use_bert}_" \
                + f"use_dte:{self.if_DTE}_" \
                + f"{self.bert_config}"
        if not os.path.exists(path):
            os.makedirs(path)
        return path + "/"

    def get_result_path(self):
        path = f"./result/{self.data_set}"
        if not os.path.exists(path):
            os.makedirs(path)
        path += f"/{self.vec_model}"
        return path + ".data"

    def load_config(self, misc_dict):
        self.word_kinds = misc_dict["word_kinds"]
        self.char_kinds = misc_dict["char_kinds"]
        self.pos_tag_kinds = misc_dict["pos_tag_kinds"]
        self.label_kinds = misc_dict["label_kinds"]
        self.id2label = misc_dict["id2label"]
        self.id2word = None

        print(self)
        self.id2word = misc_dict["id2word"]


config = Config()
