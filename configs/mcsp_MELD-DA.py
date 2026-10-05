class Param():

    def __init__(self, args):

        self.hyper_param = self._get_hyper_parameters(args)

    def _get_hyper_parameters(self, args):
        """
        Args:
            num_train_epochs (int): The number of training epochs.
            num_labels (autofill): The output dimension.
            max_seq_length (autofill): The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded.
            freeze_backbone_parameters (binary): Whether to freeze all parameters but the last layer.
            feat_dim (int): The feature dimension.
            warmup_proportion (float): The warmup ratio for learning rate.
            activation (str): The activation function of the hidden layer (support 'relu' and 'tanh').
            train_batch_size (int): The batch size for training.
            eval_batch_size (int): The batch size for evaluation.
            test_batch_size (int): The batch size for testing.
            wait_patient (int): Patient steps for Early Stop.
        """
        if args.multimodal_method == 'mcsp':
            hyper_parameters = {
                'pretrained_bert_model': 'uncased_L-12_H-768_A-12',
                'pretrain_batch_size': 128,
                'train_batch_size': 128,
                'eval_batch_size': 128,
                'test_batch_size': 128,
                'num_pretrain_epochs': 100,
                'num_train_epochs': 12,
                'pretrain': False,
                'train': True,
                'save_model': False,
                'aligned_method': 'ctc',
                'need_aligned': False,
                'freeze_pretrain_bert_parameters': True,
                'freeze_train_bert_parameters': True,
                'pretrain_temperature': 0.3,
                'train_temperature_sup': 2.5,
                'train_temperature_unsup': 1.0,
                'activation': 'tanh',
                'lr_pre': 5e-5,
                'lr': 5e-5,
                'weight_decay': 0.01,
                'feat_dim': 768,
                'hidden_size': 768,
                'grad_clip': -1.0,
                'warmup_proportion': 0.1,
                'hidden_dropout_prob': 0.01,
                'weight': 1.0,
                'base_dim': 256,
                'nheads': 8,
                'attn_dropout': 0.1,
                'relu_dropout': 0.1,
                'embed_dropout': 0.1,
                'res_dropout': 0.0,
                'attn_mask': True,
                'encoder_layers_1': 1,
                'fusion_act': 'tanh',
                'use_llm': False,
                'api_key': '',
                'tsv_path': '',
                'dataset': 'MELD-DA',
                'rep_top_k_per_cluster': 40,
                'rep_num_examples':3,
                'concept_tau': 1.0,
                'lp_edge_lambda': 3.0,
                'lp_conf_top_ratio': 0.6,
                'use_intents_as_seeds': False,
                'lp_knn_k': 10,
                'lp_tau_g': 3.5,
                'lp_batch_size': 512,
                'lp_alpha': 0.95,
                'lp_steps': 40,
            }
        else:
            print('Not Supported Multimodal Method')
            raise NotImplementedError

        return hyper_parameters
