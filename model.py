from utils import init_random_state, normalize
import torch
from torch import nn
import torch.nn.functional as F
from layers import DynamicGRU

class Fcn_net(nn.Module):
    def __init__(self, layer_sizes=None, mode='attn', pre_size = 160):
        super().__init__()
        pre_size = pre_size
        self.lin = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.act = nn.ReLU()
        for cur_size in layer_sizes:
            self.lin.append(nn.Linear(pre_size, cur_size))
            self.bn.append(nn.BatchNorm1d(num_features=cur_size,
                                            momentum=0.95,
                                            eps=0.0001))
            pre_size = cur_size
        self.out = nn.Linear(pre_size, 1)
        self.mode = mode
    def forward(self, x):
        for i in range(len(self.lin)):
            x = self.lin[i](x)
            if self.mode == 'attn':
                x = x.transpose(1, 2)
            x = self.bn[i](x)
            if self.mode == 'attn':
                x = x.transpose(1, 2)
            x = self.act(x)
        x = self.out(x)
        return x
    
class Surge(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self._build_embedding() # item embedding, ans embedding

        node_dim = config.item_embedding_dim + config.ans_embedding_dim
        self.weight_tensor  = nn.Parameter(torch.FloatTensor(
            1, node_dim)) # for metrics learning
        self.attention_mat_cluster = nn.Parameter(torch.FloatTensor(
            node_dim, node_dim))
        self.attention_mat_query = nn.Parameter(torch.FloatTensor(
            node_dim, node_dim))
        self.attention_mat_rnn = nn.Parameter(torch.FloatTensor(
            node_dim, node_dim))
        self.custom_parameter = [self.weight_tensor, self.attention_mat_cluster, self.attention_mat_query, self.attention_mat_rnn]
        self._init_parameter()
        
        self.atten_fcn = Fcn_net(
            config.att_fcn_layer_sizes, mode='attn', pre_size=node_dim * 4).cuda()  # for attention MLP
        self.aggre_lin = nn.Linear(node_dim,
                                   node_dim, bias=False)  # (4), W_a
        self.augru = DynamicGRU(input_dim=node_dim, hidden_dim=config.hidden_size)
        self.task_fcn = Fcn_net(config.layer_sizes, mode='task', pre_size=config.hidden_size + 3 * node_dim).cuda()  # task head MLP:
    
    def _init_parameter(self):
        for para in self.custom_parameter:
            nn.init.normal_(para, mean=0, std=self.config.init_value)
        nn.init.normal_(self.item_embedding.weight[1:], mean=0, std=self.config.init_value)
        nn.init.normal_(self.ans_embedding.weight[:2], mean=0, std=self.config.init_value)
        # self.item_embedding.weight.data = torch.eye(self.item_embedding.weight.shape[0]).float()
        self.ans_embedding.weight.data = torch.arange(3).unsqueeze(-1).repeat(1, self.item_embedding.weight.data.shape[1]).float()
        # self.item_embedding.weight.requires_grad_(False)
        self.ans_embedding.weight.requires_grad_(False)
        
    def _build_embedding(self):
        config = self.config
        self.item_embedding = nn.Embedding(
            config.item_num, config.item_embedding_dim, padding_idx=0)
        self.ans_embedding = nn.Embedding(3, config.ans_embedding_dim, padding_idx=2)
              # 0 false, 1 true, 2 mask

    # def _fcn_net(self, layer_sizes):
    #     fc_layers = nn.Sequential()
    #     layer_sizes = [160] + layer_sizes
    #     for i in range(len(layer_sizes) - 1):
    #         fc_layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
    #         fc_layers.append(nn.BatchNorm1d(num_features=layer_sizes[i+1],
    #                                         momentum=0.95,
    #                                         eps=0.0001))
    #         fc_layers.append(nn.ReLU())
    #     fc_layers.append(nn.Linear(layer_sizes[-1], 1))
    #     return fc_layers

    def _interset_graph(self, X): # ok
        '''
        build graph based on metrics learning 
        return min-max A
        '''
        # Node similarity metric learning
        
        X_fts = X * self.weight_tensor
        X_fts = F.normalize(X_fts, dim=2)
        S_one = torch.bmm(X_fts, X_fts.transpose(1, 2)) # sym
        S_min = torch.min(S_one, dim=-1, keepdim=True)[0]
        S_max = torch.max(S_one, dim=-1, keepdim=True)[0]
        S_one = (S_one - S_min) / (S_max - S_min)  # (B, max_step, max_step)
        S_one = torch.nan_to_num(S_one, nan=0.0)
        return S_one

    def create_mask(self, seq_lengths, max_len): # ok
        batch_size = seq_lengths.size(0)
        mask = torch.arange(max_len).to(seq_lengths.device).unsqueeze(0).repeat(
            batch_size, 1) < seq_lengths.unsqueeze(1)
        return mask.float()

    def graph_sparsification(self, S, relative_threshold=None, to_keep_edge=None): #ok
        # S (bs, max_step, max_step)
        # for each batch, reserve largets  (ratio * num_edge) , return mask 
        S_flatten = S.view(S.shape[0], -1) #(bs, max_step * max_step)
        sorted_S_flatten, _ = S_flatten.sort(dim=-1, descending=True)
        num_edges = sorted_S_flatten.count_nonzero(-1) #(bs, )
        if relative_threshold:
            to_keep_edge = torch.ceil(
                num_edges * relative_threshold).to(torch.long)
        threshold_score = sorted_S_flatten[range(
            S_flatten.shape[0]), to_keep_edge] # (bs, )
        A = (S_flatten > threshold_score.reshape(-1, 1)
             ).float().reshape(S.shape) #(bs, max_step, max_step)
        return A

    def _attention_fcn(self, query, key_value, name='cluster', return_alpha=True):
        """Apply attention by fully connected layers.

        Args:
            query (obj): The embedding of target item or cluster which is regarded as a query in attention operations.
            key_value (obj): The embedding of history items which is regarded as keys or values in attention operations.
            return_alpha (obj): Returning attention weights

        Returns:
            output (obj): Weighted sum of value embedding.
            att_weights (obj):  Attention weights
        """
        if name == 'cluster': # ok
            att_inputs = torch.matmul(key_value, self.attention_mat_cluster)
        elif name == 'query': # ok
            att_inputs = torch.matmul(key_value, self.attention_mat_query)
        elif name == 'AUGRU':
            att_inputs = torch.matmul(key_value, self.attention_mat_rnn)
        
        if len(query.shape) != len(att_inputs.shape): # for target
            query = query.unsqueeze(1).repeat(1, att_inputs.shape[1], 1)
        last_hidden_nn_layer = torch.concat(
            [att_inputs, query, att_inputs - query, att_inputs * query], -1) #(bs, max_step, h * 4)
        # import ipdb; ipdb.set_trace()
        att_fnc_output = self.atten_fcn(last_hidden_nn_layer).squeeze()
        mask_paddings = torch.ones_like(att_fnc_output) * float('-inf')
        att_weights = F.softmax(
            torch.where(self.mask > 0, att_fnc_output, mask_paddings),
            dim=-1
        )
        output = key_value * att_weights.unsqueeze(-1)
        if not return_alpha:
            return output
        else:
            return output, att_weights  # , (bs, max_step)

    def _interest_fusion_extraction(self, X, A):
        '''
        Interest fusion and extraction via graph convolution and graph pooling 

        Args:
            X (obj): Node embedding of graph (bs, max_step, h_pro + h_ans)
            A (obj): Adjacency matrix of graph (bs, max_step, max_step)

        Returns:
            X (obj): Aggerated cluster embedding 
            A (obj): Pooled adjacency matrix 
            graph_readout (obj): Readout embedding after graph pooling
            cluster_score (obj): Cluster score for AUGRU in prediction layer
        '''
        B, L, L = A.shape
        A_bool = (A > 0).float()
        A_bool = A_bool * (torch.ones(L, L, device=A_bool.device, dtype=torch.float32) - 
                           torch.eye(L, device=A_bool.device, dtype=torch.float32)) + torch.eye(L, device=A_bool.device, dtype=torch.float32)

        degrees = torch.sum(A_bool, dim=-1)
        D_inv_sqrt = torch.diag_embed(torch.pow(degrees, -0.5))
        A = torch.matmul(torch.matmul(D_inv_sqrt, A_bool), D_inv_sqrt) #(bs, L, L)
        X_q = torch.matmul(A, torch.matmul(A, X))  # B*L*F, (6), h_ic
        # cluster- and query-aware attention
        _, f_1 = self._attention_fcn(X_q, X, name='cluster')
        _, f_2 = self._attention_fcn(
            self.target_pro_embedding, X, name='query')

        # graph attentive convolution
        # (bs, max_step, max_step)
        
        E = A_bool * f_1.unsqueeze(1) + A_bool * \
            f_2.unsqueeze(1).permute(0, 2, 1) #(bs, max_step, max_step)
        E = F.leaky_relu(E)
        mask_paddings = torch.ones_like(E) * (float('-inf'))
        
        E = F.softmax(
            torch.where(A_bool > 0, E, mask_paddings),
            dim=-1
        )  # (8)
        Xc_one = torch.matmul(E, X) # (B, L, L) x (B, L, F)

        Xc_one = self.aggre_lin(Xc_one) + X
        Xc = F.leaky_relu(Xc_one)
        # ========================== above ok ==================================

        # interest extraction
        # cluster fitness score
        # (bs, max_step, h_pro + h_ans)(11)
        ###############################################################################
        X_q = torch.matmul(A, torch.matmul(A, Xc))
        _, f_1 = self._attention_fcn(X_q, Xc, name='cluster')
        _, f_2 = self._attention_fcn(
            self.target_pro_embedding, Xc, name='query')
        cluster_score = f_1 + f_2
        # import ipdb; ipdb.set_trace()
        mask_paddings = torch.ones_like(cluster_score) * (-(2 ** 32) + 1)
        cluster_score = F.softmax(
            torch.where(self.mask > 0, cluster_score, mask_paddings),
            dim=-1
        )  # （bs, max_step)
        ###############################################################################
        # graph pooling
        num_nodes = torch.sum(self.mask, 1)  # (bs, )
        
        to_keep = torch.where(
            num_nodes > self.config.pool_length,
            self.config.pool_length,
            num_nodes
        ).long()
        assert to_keep.min() > 0
        cluster_score = cluster_score * self.mask
        self.mask = self.graph_sparsification(
            cluster_score, to_keep_edge=to_keep)
        assert self.mask.sum(dim=-1).min() > 0, self.mask
        self.reduced_sequence_length = torch.sum(self.mask, 1)

        # ensure graph connectivity
        E = E * self.mask.unsqueeze(-1) * self.mask.unsqueeze(-2)
        A = torch.matmul(torch.matmul(E, A_bool), E.permute(0, 2, 1))
        graph_readout = torch.sum(
            Xc * cluster_score.unsqueeze(-1) * self.mask.unsqueeze(-1), 1)
        return Xc, A, graph_readout, cluster_score

    def flatten(self, X, alphas):
        # flatten pooled graph to reduced sequence
        sorted_mask, sorted_mask_index = torch.sort(
            self.mask, dim=-1, descending=True)  # B*L -> B*L
        # (bs, max_step, h_pro + h_ans)
        X = torch.gather(
            X, dim=1, index=sorted_mask_index.unsqueeze(-1).expand(-1, -1, X.size(-1)))
        self.mask = sorted_mask
        self.reduced_sequence_length = torch.sum(self.mask, 1)  # (bs, )
        assert self.reduced_sequence_length.min() > 0
        # cut useless sequence tail per batch
        self.to_max_length = torch.max(self.reduced_sequence_length).long()
        X = X[:, :self.to_max_length, :]
        self.mask = self.mask[:, :self.to_max_length]
        self.reduced_sequence_length = torch.sum(self.mask, 1).long()
        # assert self.reduced_sequence_length.min() > 0, self.reduced_sequence_length
        # use cluster score as attention weights in AUGRU
        _, alphas = self._attention_fcn(self.target_pro_embedding, X, 'AUGRU')
        rnn_output = self.augru(X, alphas)
        
        final_state = rnn_output[range(
            rnn_output.shape[0]), self.reduced_sequence_length-1, :]
        return final_state

    def forward(self, *data):
        # (bs, max_step), (bs, max_step), (bs, ), (bs, ), (bs, )
        his_pro, his_y, his_len, cur_pro, cur_y = data
        B = his_len.shape[0]
        self.mask = self.create_mask(his_len, his_pro.shape[1])  # (bs, max_step)
        his_pro_embedding = self.item_embedding(
            his_pro)  # (bs, max_step, h_pro)
        his_ans_embedding = self.ans_embedding(his_y)
        self.target_pro_embedding = self.item_embedding(cur_pro)  # (bs, h_pro)
        self.target_pro_embedding = torch.concat([self.target_pro_embedding, torch.zeros(B, self.config.ans_embedding_dim).to(self.target_pro_embedding.device)], dim = -1)
        # (bs, max_step, h_pro, h_ans)
        X = torch.concat([his_pro_embedding, his_ans_embedding], dim=2)
        #=============================above ok =================================
        # 1. Interest graph: Graph construction based on metric learning
        S = self._interset_graph(X)
        # mask invalid nodes
        # min-max masks padding -> min
        S = S * self.mask.unsqueeze(-1) * self.mask.unsqueeze(-2) # (bs, max_step, max_step), (bs, max_step, 1), (bs, 1, max_step)
        #=============================above ok =================================
        # Graph sparsification via seted sparseness
        A = self.graph_sparsification(S, relative_threshold=self.config.relative_threshold)
        #=============================above ok =================================
        # import ipdb; ipdb.set_trace()
        # 2. interset fusion and extraction
        X, A, graph_readout, alphas = self._interest_fusion_extraction(X, A)
        # flatten sequence -> augru -> final state
        final_state = self.flatten(X, alphas)
        self.augru(X, alphas)
        model_output = torch.concat(
            [final_state, graph_readout, self.target_pro_embedding, self.target_pro_embedding*graph_readout], 1)
        logit = self.task_fcn(model_output)
        return logit.squeeze()
