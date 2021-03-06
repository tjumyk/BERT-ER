import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer
from transformer_encoder import MultiHeadedAttention, attention, Transformer, clones
from cross_utils import Cross_Encoding, Orthogonal_matrix
from Mybert import BertModel
import time


def stringlize_batch(batch, doc, pairs=False):
    if pairs is True:
        batch_size = len(batch[0])
        str_batch = [[" ".join(doc[str(batch[0][i].item())]), " ".join(doc[str(batch[1][i].item())])] for i in range(batch_size)]
    else:
        batch_size = len(batch)
        str_batch = [" ".join(doc[str(batch[i].item())]) for i in range(batch_size)]
    return str_batch


def bert_prepare(batch, tokenizer, device, MAX_LEN=100):
    encoded_dict = tokenizer.batch_encode_plus(
        batch,  # Sentence to encode.
        add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
        #add_special_tokens=False,  # Add '[CLS]' and '[SEP]'
        max_length=MAX_LEN,  # Pad & truncate all sentences.
        pad_to_max_length=True,
        return_attention_mask=True,  # Construct attn. masks.
        return_tensors='pt',  # Return pytorch tensors.
        return_token_type_ids=True,
    )
    return encoded_dict['input_ids'].to(device), encoded_dict['attention_mask'].to(device), encoded_dict['token_type_ids'].to(device)

def _generate_attribute_mask(tokenizer, str_att, MAX_LEN):
    attribute_mask = [0 for _ in range(MAX_LEN)]
    offset = 0
    for i in range(len(str_att)):
        encoded_dict = tokenizer.batch_encode_plus(
            str_att[i],  # Sentence to encode.
            add_special_tokens=False,  # Add '[CLS]' and '[SEP]'
            max_length=MAX_LEN,  # Pad & truncate all sentences.
            pad_to_max_length=True,
            return_attention_mask=True,  # Construct attn. masks.
            return_tensors='pt',  # Return pytorch tensors.
            return_token_type_ids=False,
        )
        enc = encoded_dict['attention_mask']
        count = torch.sum(enc, dim=1)
        offset += 1
        for c in range(len(str_att[i])):
            n = count[c].item()
            for j in range(offset, min(offset + n, MAX_LEN)):
                attribute_mask[j] = c + 1
            offset += n
    return torch.tensor(attribute_mask).unsqueeze(dim=0)

def bert_prepare_from_raw_dataset(tokenizer, batch, doc, device, MAX_LEN, Pair=True):
    if Pair is True:
        batch_size = len(batch[0])
        str_att = [[[" ".join(x) for x in doc[str(batch[i][c].item())]] for i in range(2)] for c in range(batch_size)]
        str_batch = [[" ".join(str_att[c][i]) for i in range(2)] for c in range(batch_size)]
    else:
        batch_size = len(batch)
        str_att = [[[" ".join(x) for x in doc[str(batch[c].item())]]] for c in range(batch_size)]
        str_batch = [" ".join(str_att[c][0]) for c in range(batch_size)]

    attribute_ids = [_generate_attribute_mask(tokenizer, str_att[i], MAX_LEN) for i in range(batch_size)]
    attribute_ids = torch.cat(attribute_ids, dim=0).to(device)

    input_ids, attention_masks, type_ids = bert_prepare(str_batch, tokenizer, device, MAX_LEN)
    return input_ids, attention_masks, type_ids, attribute_ids

class BERT_full(nn.Module):
    def __init__(self, pretrained_model, args):
        super(BERT_full, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(pretrained_model, do_lower_case=False)
        self.model = BertModel.from_pretrained(pretrained_model, output_hidden_states=True, output_attentions=True)
        self.MAX_LEN = args.max_len
        self.softmax = nn.LogSoftmax(dim=1)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.attentions = clones(Cross_Encoding(768), 13)
        self.kernel_num = 64
        self.filter_sizes = [1, 2]
        self.linear = nn.Linear(self.kernel_num*len(self.filter_sizes)*2*1, 2)
        self.encoders = nn.ModuleList([torch.nn.Conv1d(in_channels=768*3,
                                                       out_channels=self.kernel_num,
                                                       padding=0,
                                                       kernel_size=filter_size) for i, filter_size in enumerate(self.filter_sizes)])
        self.cnn = clones(self.encoders, 13)
        self.bn = nn.ModuleList(nn.BatchNorm1d(self.kernel_num) for _ in range(13))

        #self.dsh_linear = nn.Linear(768, args.bits, bias=False)
        self.isEval = False
        #self.or_matrix = Orthogonal_matrix(768, args.bits)
        self.or_matrix = nn.Parameter(torch.zeros([768, args.bits], dtype=torch.float, requires_grad=True), requires_grad=True)
        nn.init.xavier_uniform_(self.or_matrix, gain=1)
        self.encode_end = 0
        self.encode_start = 0
        self.block_end = 0
        self.block_start = 0
        self.interaction_start = 0
        self.interaction_end = 0
        self.or_matrix1 = []

    def dsh_linear(self, m):
        if self.isEval == False:
            #return torch.mm(m, self.or_matrix())
            return torch.mm(m, self.or_matrix)
        else:
            return torch.mm(m, self.or_matrix1)

    def fix_hyperplanes(self):
        self.or_matrix1 = self.or_matrix()

    def _get_encoding(self, batch, doc):
        input_ids, attention_masks, type_ids, attribute_ids = bert_prepare_from_raw_dataset(self.tokenizer, batch, doc,
                                                                                            self.device, self.MAX_LEN,
                                                                                            False)
        encoded_layers, _, hidden_states, attentions = self.model(input_ids, attention_mask=attention_masks, token_type_ids=type_ids,
                                       attribute_ids=attribute_ids)
        return hidden_states, attention_masks

    def _cross_encoding(self, Q, A, mask):
        EQ = []
        EA = []
        for l, q, a in zip(self.attentions, Q, A):
            eq, ea = l(q, a, mask)
            EQ.append(eq)
            EA.append(ea)
        return EQ, EA

    def _cnn_encoder(self, FQ, FA, Q_mask, A_mask):
        def encode(encoders, x, norm1):
            enc_outs = []
            for encoder in encoders:
                f_map = encoder(x.transpose(-1, -2))
                enc_ = F.relu(norm1(f_map))
                k_h = enc_.size()[-1]
                enc_ = F.max_pool1d(enc_, kernel_size=k_h)
                enc_ = enc_.squeeze(dim=-1)
                enc_outs.append(enc_)
            return torch.cat(enc_outs, dim=1)

        l_enc_outs = []
        for l, q, a, norm in zip(self.cnn, FQ, FA, self.bn):
            if Q_mask is not None:
                q.masked_fill_(Q_mask == 0, 0)
            rq = encode(l, q, norm)
            if A_mask is not None:
                a.masked_fill_(A_mask == 0, 0)
            ra = encode(l, a, norm)
            l_enc_outs.append(torch.cat((rq, ra), dim=1))

        return l_enc_outs

    def _get_features(self, Q, EQ, A, EA):
        re_q = []
        re_a = []
        for q, eq in zip(Q, EQ):
            sub = q - eq
            sub_q = torch.mul(sub, sub)
            mul_q = torch.mul(q, eq)
            re_q.append(torch.cat((q, sub_q, mul_q), dim=-1))

        for a, ea in zip(A, EA):
            sub = a - ea
            sub_a = torch.mul(sub, sub)
            mul_a = torch.mul(a, ea)
            re_a.append(torch.cat((a, sub_a, mul_a), dim=-1))
        return re_q, re_a

    def forward(self, batch, doc):
        self.encode_start += time.time()
        hs_q, Q_mask = self._get_encoding(batch[0], doc)
        Q = hs_q[2:3]
        hs_a, A_mask = self._get_encoding(batch[1], doc)
        A = hs_a[2:3]
        Q_Amask_matrix = torch.matmul(Q_mask.unsqueeze(dim=-1).float(), A_mask.unsqueeze(dim=-2).float()).byte()
        Q_mask = Q_mask.unsqueeze(dim=-1).byte()
        A_mask = A_mask.unsqueeze(dim=-1).byte()
        self.encode_end += time.time()
        self.interaction_start += time.time()
        EQ, EA = self._cross_encoding(Q, A, Q_Amask_matrix)
        FQ, FA = self._get_features(Q, EQ, A, EA)
        encoding = self._cnn_encoder(FQ, FA, Q_mask, A_mask)
        re = torch.cat(encoding, dim=1)
        score = F.relu(self.linear(re))
        match_predict = self.softmax(score)
        self.interaction_end += time.time()
        self.block_start += time.time()
        hash_code_q = self.dsh_linear(hs_q[12][:, 0, :])
        hash_code_a = self.dsh_linear(hs_a[12][:, 0, :])
        self.block_end += time.time()
        return hash_code_q, hash_code_a, match_predict