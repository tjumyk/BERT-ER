import torch.utils.data as Data
import csv
from bert import *
import bert
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import random
import copy
from Hamming_dict import Hamming_dict

def get_model_class(model_name, pretrained_model, args):
    model_list = ['BERT_cross_encoding', 'BERT_sbert', 'BERT_pooling', 'BERT_use_cls', 'BERT_sequence_classification_head', 'BERT_attribute_mask', 'BERT_transformer_encoding', 'BERT_neg_att', 'BERT_delta', 'BERT_DSH', 'BERT_full']
    for x in model_list:
        if x.find(model_name) != -1:
            AClass = getattr(bert, x)(pretrained_model, args)
    return AClass


def load_tdt(info, read_from_csv=False):
    if read_from_csv is True:
        train = info[0]
        dev = info[1]
        test = info[2]
    else:
        train = info['data']['train']
        dev = info['data']['dev']
        test = info['data']['test']
    smap = {}
    tmap = {}
    def build_stmap(mapping, smap, tmap):
        for (x, y, l) in mapping:
            if l != 1:
                continue
            if x not in smap:
                smap[x] = set()
            smap[x].add(y)
            if y not in tmap:
                tmap[y] = set()
            tmap[y].add(x)
    build_stmap(train, smap, tmap)
    build_stmap(dev, smap, tmap)
    build_stmap(test, smap, tmap)
    return train, dev, test, smap, tmap


def train_batch(train, batch_number, isTest = False):
    S = []
    T = []
    L = []
    for (x,y,l) in train:
        S.append(x)
        T.append(y)
        if l == -1 or l == 0:
            L.append(0)
        else:
            L.append(1)
    S = torch.LongTensor(S)
    T = torch.LongTensor(T)
    L = torch.LongTensor(L)
    data = Data.TensorDataset(S, T, L)
    if isTest == True:
        return S, T, L
    loader = Data.DataLoader(
        dataset=data,
        batch_size=batch_number,
        shuffle=True,
        num_workers=1,
    )
    return loader


def compute_f1(tp, tn, fp, fn):
    p = 0
    r = 0
    f1 = 0
    if tp + fp != 0:
        p = tp / (tp + fp)
    if tp + fn != 0:
        r = tp / (tp + fn)
    if p + r != 0:
        f1 = 2 * p * r / (p + r)
    print("tp,tn,fp,fn:", tp, tn, fp, fn)
    return p, r, f1

def instance_to_str(x, y, doc_content):
    A = doc_content[str(x.item())]
    B = doc_content[str(y.item())]
    str1 = "P: "
    str1 += " ".join(A)
    str1 += " "
    str2 = " Q: "
    str2 += " ".join(B)
    str2 += " "
    return str1 + str2 + '\n'

def validation(bertm, dev, doc_att, device, log=True):
    values = [0, 0, 0, 0]  # tp tn fp fn
    loader = train_batch(dev, 32)
    bertm.eval()
    f = open("wrong cases.txt", 'w', encoding='utf-8')
    with torch.no_grad():
        for step, batch in enumerate(loader):
            batch_l = batch[2]
            predict = bertm.forward(batch, doc_att)
            _, indices = torch.max(predict, dim=1)
            for i in range(len(indices)):
                if indices[i] == 1:
                    if batch_l[i] == 1:
                        values[0] += 1
                    else:
                        values[2] += 1
                        if log is True:
                            f.write('[fp] ' + instance_to_str(batch[0][i], batch[1][i], doc_att))
                elif batch_l[i] == 1:
                    values[3] += 1
                    if log is True:
                        f.write('[fn] ' + instance_to_str(batch[0][i], batch[1][i], doc_att))
                else:
                    values[1] += 1
    p, r, f1 = compute_f1(values[0], values[1], values[2], values[3])
    f.write(str(p) + str(r) + str(f1) + '\n')
    f.flush()
    f.close()
    return p, r, f1

def readData(dataset, f_model=False, ratio=10):
    file_a = "Structured/" + dataset + "/tableA.csv"
    file_b = "Structured/" + dataset + "/tableB.csv"
    if f_model is False:
        file_train = "Structured/" + dataset + "/train.csv"
        file_dev = "Structured/" + dataset + "/valid.csv"
        file_test = "Structured/" + dataset + "/test.csv"
    else:
        file_mapping = "Structured/" + dataset + "/mapping.csv"

    def remove_stopwords(sent):
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(sent)
        filtered_sentence = [w for w in word_tokens if not w in stop_words]
        return filtered_sentence

    def readTable(file, f_model=False):
        data = {}
        if f_model is False:
            with open(file, encoding='utf-8') as f:
                reader = csv.reader(f)
                id = 0
                for r in reader:
                    data[r[0]] = [remove_stopwords(sent) for sent in r[1:]]
                    #for sent in r[1:]:
                    #    fsent = remove_stopwords(sent)
                    #    data[r[0]].append(fsent)
                if 'id' in data:
                    del (data['id'])
            return data
        else:
            key_convert = {}
            with open(file, encoding='utf-8') as f:
                reader = csv.reader(f)
                cid = 0
                for r in reader:
                    key_convert[r[0]] = str(cid)
                    data[str(cid)] = [remove_stopwords(sent) for sent in r[1:]]
                    cid = cid + 1
                if '\ufeffid' in key_convert:
                    del (key_convert['\ufeffid'])
                    del (data['0'])
            return data, key_convert


    def readMapping(file, kca=None, kcb=None, f_model=False):
        mapping = []
        if f_model is False:
            with open(file) as f:
                reader = csv.reader(f)
                for r in reader:
                    mapping.append((r[0], r[1], r[2]))
        else:
            map_dict = {}
            with open(file) as f:
                reader = csv.reader(f)
                for r in reader:
                    if r[0] not in kca or r[1] not in kcb:
                        continue
                    r0 = kca[r[0]]
                    r1 = kcb[r[1]]
                    mapping.append((r0, r1, '1'))
                    if r0 not in map_dict:
                        map_dict[r0] = [r1]
                    else:
                        map_dict[r0].append(r1)
            #del (mapping[0])
            return mapping, map_dict
        del (mapping[0])
        return mapping

    def check(mapping, ta, tb):
        for x in mapping:
            if x[0] not in ta or x[1] not in tb:
                mapping.remove(x)
                print("!!!")
        # for i in range(len(mapping)):
        #     if mapping[i][0] not in ta or mapping[i][1] not in tb:
        #         mapping.pop(i)
        #         print("!!!")


    def generate_negative_examples(ta, tb, mapping_dict, ratio):
        batch_dict = copy.deepcopy(mapping_dict)
        total_batch = []
        size = ratio * len(mapping_dict)
        print(size)
        def generate_next_batch():
            batch = []
            rta = list(ta.keys())
            rtb = list(tb.keys())
            random.shuffle(rta)
            random.shuffle(rtb)
            for r1, r2 in zip(rta, rtb):
                if r1 not in batch_dict:
                    batch.append((r1, r2, '0'))
                    batch_dict[r1] = [r2]
                else:
                    if r2 not in batch_dict[r1]:
                        batch.append((r1, r2, '0'))
                        batch_dict[r1].append(r2)
            return batch
        while len(total_batch) < size:
            batch = generate_next_batch()
            r = len(batch) - size + len(total_batch)
            if r < 0:
                total_batch.extend(batch)
            else:
                total_batch.extend(batch[0:len(batch)-r])
        return total_batch

    def split_dataset(dataset):
        train_size = int(len(dataset)*0.8)
        dev_size = int(len(dataset) * 0.0)
        #test_size = len(dataset) - train_size - dev_size
        random.shuffle(dataset)
        return dataset[0:train_size], dataset[train_size:dev_size], dataset[train_size+dev_size:]

    if f_model is False:
        table_a = readTable(file_a)
        table_b = readTable(file_b)
        train = readMapping(file_train)
        dev = readMapping(file_dev)
        test = readMapping(file_test)
        check(train, table_a, table_b)
        check(dev, table_a, table_b)
        check(test, table_a, table_b)
    else:
        table_a, key_convert_a = readTable(file_a, f_model)
        table_b, key_convert_b = readTable(file_b, f_model)
        positive, mapping_dict = readMapping(file_mapping, key_convert_a, key_convert_b, f_model)
        check(positive, table_a, table_b)
        negative = generate_negative_examples(table_a, table_b, mapping_dict, ratio)
        positive.extend(negative)
        train, dev, test = split_dataset(positive)
    return table_a, table_b, train, dev, test

def merge_table(ta, tb):
    all = ta.copy()
    offset = len(ta)
    for (id,value) in tb.items():
        new_id = offset + int(id)
        all[str(new_id)] = value
    return offset, all

def convert_mapping(mapping, offset, cosine_loss = False):
    new_mapping = []
    for (x, y, l) in mapping:
        new_y = int(y) + offset
        if cosine_loss == True:
            if l == '0':
                l = '-1'
        new_mapping.append((int(x), new_y, int(l)))
    return new_mapping


def load_from_csv(dataset, f_model=False, ratio=10):
    table_a, table_b, train, dev, test = readData(dataset, f_model, ratio)
    offset, all_doc = merge_table(table_a, table_b)
    train = convert_mapping(train, offset)
    dev = convert_mapping(dev, offset)
    test = convert_mapping(test, offset)
    #table_a.update(table_b)
    #all_doc = table_a
    torch.save((all_doc, train, dev, test), "Structured/" + dataset + "/data.pkl")
    return all_doc, train, dev, test


def data_for_blocking(dataset):
    ## this is used on after-blocking dataset to gnereate extra negative examples, not for genuines full dataset
    def generate_training_data(train, dev, test, lenA, lenB, num):
        def add2dic(dic, train):
            for (x,y,l) in train:
                if l == '1':
                    if int(x) not in dic:
                        dic[int(x)] = []
                    dic[int(x)].append(int(y))

        pos_dic = {}
        add2dic(pos_dic, train)
        add2dic(pos_dic, dev)
        add2dic(pos_dic, test)

        new_data = []
        while len(new_data) < num:
            ra = random.randint(0, lenA-1)
            rb = random.randint(0, lenB-1)
            if ra == rb:
                continue
            if ra in pos_dic:
                if rb in pos_dic[ra]:
                    continue
            new_data.append((str(ra), str(rb), '0'))
        return new_data

    table_a, table_b, train, dev, test = readData(dataset)
    offset, all_doc = merge_table(table_a, table_b)
    neg_train = generate_training_data(train, dev, test, len(table_a), len(table_b), 10000)
    neg_train = convert_mapping(neg_train, offset)
    train = convert_mapping(train, offset)
    dev = convert_mapping(dev, offset)
    test = convert_mapping(test, offset)
    new_train = neg_train
    new_train.extend(train)
    new_train.extend(dev)
    new_train.extend(test)

    return all_doc, new_train


def hashing_loss(Q, A, cls, W, m=16, alpha=0.01):
    """
    compute hashing loss
    automatically consider all n^2 pairs
    """
    dist = ((Q - A) ** 2).sum(dim=1)
    reg = (Q.abs() - 1).abs().sum(dim=1) + (A.abs() - 1).abs().sum(dim=1)
    y = cls.float()
    loss = (y / 2) * dist + ((1 - y) / 2) * (m - dist).clamp(min=0)
    I = torch.eye(768).to('cuda')
    rm = torch.sum(torch.abs(I - torch.mm(W, W.t())), (-1, -2))

    loss = loss + alpha * reg + alpha * rm
    return loss.mean()



def validation_blocking(bertm, dev, doc_att):
    DB = 0  # DB: detectable duplicates
    B = 0  # B:cardinality of buckets
    DE = 0  # all detectable duplicates
    E = len(dev) ** 2  # total cardinality
    buckets = {}
    multi = {}

    def _get_hash_code(Q, A):
        hc_q = torch.sign(Q)
        hc_a = torch.sign(A)
        return hc_q, hc_a

    def _build_bucket(hc, buckets, multi):
        for c in hc.tolist():
            c = [int(x) for x in c]
            sc = ''.join(map(str, c))
            if sc not in buckets:
                buckets[sc] = []
                multi[sc] = 1
            else:
                multi[sc] = multi[sc] + 1
        return buckets, multi

    def _insert_bucket(hc, buckets):
        for c in hc.tolist():
            c = [int(x) for x in c]
            sc = ''.join(map(str, c))
            if sc in buckets:
                buckets[sc].append(sc)
        return buckets

    def accumulate_DB(hc_q, hc_a, batch_l, DB, DE):
        for q, a, l in zip(hc_q.tolist(), hc_a.tolist(), batch_l):
            if l == 1:
                sq = ''.join(map(str, q))
                sa = ''.join(map(str, a))
                DE = DE + 1
                if sq == sa:
                    DB = DB + 1
        return DB, DE

    def compute_B(buckets, multi):
        count = 0
        for key in buckets.keys():
            count += len(buckets[key]) * multi[key]
        return count

    loader = train_batch(dev, 32)
    bertm.eval()
    with torch.no_grad():
        for step, batch in enumerate(loader):
            batch_l = batch[2]
            Q, A = bertm.forward(batch, doc_att)
            hc_q, hc_a = _get_hash_code(Q, A)
            buckets, multi = _build_bucket(hc_q, buckets, multi)
            buckets = _insert_bucket(hc_a, buckets)
            DB, DE = accumulate_DB(hc_q, hc_a, batch_l, DB, DE)
        B = compute_B(buckets, multi)
        RR = B/E
        PC = DB/DE
    return RR, PC


def validation_full(bertm, dev, doc_att, d):
    def _get_hash_code(Q, A):
        hc_q = torch.clamp(torch.sign(Q), min=0.0)
        hc_a = torch.clamp(torch.sign(A), min=0.0)
        return hc_q, hc_a

    def accumulate_DB(hc_q, hc_a, batch_l, DB, DE, d):
        DE += torch.sum(batch_l, dim=0).item()
        Hamming_d = torch.sum((hc_q - hc_a).abs(), dim=1)
        valide_pos = torch.clamp(torch.sign(Hamming_d - d - 1), max=0.0).abs().unsqueeze(dim=1)
        DB += torch.mm(valide_pos.t(), batch_l.float().cuda().unsqueeze(dim=1)).item()
        return DB, DE

    loader = train_batch(dev, 32)
    bertm.eval()

    # blocking
    DB = 0  # DB: detectable duplicates
    #B = 0  # B:cardinality of buckets
    DE = 0  # all detectable duplicates
    E = len(dev) ** 2  # total cardinality
    buckets = Hamming_dict(d)

    # matching
    values = [0, 0, 0, 0]  # tp tn fp fn

    with torch.no_grad():
        for step, batch in enumerate(loader):
            batch_l = batch[2]
            Q, A, predict = bertm.forward(batch, doc_att)

            # blocking part
            hc_q, hc_a = _get_hash_code(Q, A)
            buckets.build_bucket(hc_q)
            buckets.insert_bucket(hc_a)
            DB, DE = accumulate_DB(hc_q, hc_a, batch_l, DB, DE, d)

            # matching part
            _, indices = torch.max(predict, dim=1)
            for i in range(len(indices)):
                if indices[i] == 1:
                    if batch_l[i] == 1:
                        values[0] += 1
                    else:
                        values[2] += 1
                elif batch_l[i] == 1:
                    values[3] += 1
                else:
                    values[1] += 1
        p, r, f1 = compute_f1(values[0], values[1], values[2], values[3])
        B = buckets.compute_B()
        RR = B / E
        PC = DB / DE
    return RR, PC, p, r, f1

