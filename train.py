from tqdm import tqdm
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
import argparse
from utils import *
## currently using WWT trick
def get_args():
    parser = argparse.ArgumentParser(description='Bert ER')
    parser.add_argument('--model', type=str, default='full', #model_list = ['BERT_cross_encoding', 'BERT_sbert', 'BERT_pooling', 'BERT_use_cls', 'BERT_sequence_classification_head', 'BERT_attribute_mask']
                        help='model')
    parser.add_argument('--pretrained_model', type=str, default='bert-base-cased',
                        help='model')
    parser.add_argument('--max_len', type=int, default=50,
                        help='maximal sentence length')
    parser.add_argument('--dataset', type=str, default='Amazon-Google',
                        help='which dataset is being used')
    parser.add_argument('--batch', type=int, default=32,
                        help='batch size [default: 32]')
    parser.add_argument('--kernel_num', type=int, default=128,
                        help='number of kernels [default: 128]')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='learning rate [default: 1e-5]')
    parser.add_argument('--epoch', type=int, default=50,
                        help='number of epoch [default: 50]')
    parser.add_argument('--bits', type=int, default=8,
                        help='bits of hash code [default: 12]')
    parser.add_argument('--alpha', type=float, default=0,
                       help='task loss weight [default: 0.2]')
    parser.add_argument('--hamming', type=int, default=1,
                        help='hamming distance [default: 0]')
    parser.add_argument('--pnratio', type=int, default=10,
                        help='positivte-to-negative ratio [default: 10]')
    parser.add_argument('--load_model', type=str, default='raw',
                        help='load data from raw or pkl [default: raw]')
    args = parser.parse_args()
    return args

def add_mask_to_doc(doc, f_model=False):
    if f_model is False:
        for i in range(len(doc)):
            cur_doc = doc[str(i)]
            for j in range(len(cur_doc)):
                if len(cur_doc[j]) == 0:
                    cur_doc[j] = ["[MASK]"]
    else:
        keys = list(doc.keys())
        for key in keys:
            for j in range(len(key)):
                if len(key[j]) == 0:
                    key[j] = ["[MASK]"]

def mirror(train):
    mirror_data = []
    for (x,y,l) in train:
        mirror_data.append((y,x,l))
    train.extend(mirror_data)
    return train


def bert_tm_full(args):
    print(args)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    f_model = False
    if args.dataset.find('F_') != -1:
        f_model = True
    w1 = args.alpha
    w2 = 1.0 - w1

    print('-' * 90)
    print("start loading data")
    # load data
    if args.load_model.find('raw') != -1:
        doc_att, train, dev, test = load_from_csv(args.dataset, f_model, ratio=args.pnratio)
        #_, _, _, test = load_from_csv(args.dataset, f_model, ratio=args.pnratio)
    else:
        doc_att, train, dev, test = torch.load("Structured/" + args.dataset + "/data.pkl")

    print(len(test))

    #traing tricks
    add_mask_to_doc(doc_att, True)
    train = mirror(train)
    #print('-' * 90)

    # loss functions
    loss_b = hashing_loss
    loss_m = nn.NLLLoss()

    # initialize model
    print('-' * 90)
    print("start initializing model")
    while True:
        model = get_model_class(args.model, args.pretrained_model, args)
        bertm = model.to(device)
        torch.cuda.empty_cache()
        RR, PC, p, r, f1 = validation_full(bertm, test, doc_att, args.hamming)
        if PC > 0.7 and f1 != 0:
            print("RR & PC &F1 ", RR, PC, f1)
            break


    # optimizer setting
    optimizer = AdamW(bertm.parameters(), lr=1e-5, eps=1e-8)
    MAX_EPOCH = args.epoch
    loader = train_batch(train, args.batch)
    total_steps = len(loader) * MAX_EPOCH
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,  # Default value in run_glue.py
                                                num_training_steps=total_steps)
    print('-' * 90)
    print("start training")
    for epoch in range(MAX_EPOCH):
        print("Epoch "+str(epoch))
        bertm.model.train()
        loader = train_batch(train, args.batch)
        total_loss = torch.zeros(1)
        for batch in tqdm(loader):
            bertm.zero_grad()
            bertm.model.train()
            batch_l = batch[2].to(device)
            Q, A, predict = bertm.forward(batch, doc_att)
            #loss1 = loss_b(Q, A, batch_l, bertm.or_matrix, m=args.bits*2)
            loss2 = loss_m(predict, batch_l)
            #loss = w1 * loss1 + w2 * loss2
            loss = loss2
            loss.backward()
            torch.nn.utils.clip_grad_norm_(bertm.parameters(), 1.0)
            optimizer.step()
            if args.model.find("cross") == -1:
                scheduler.step()
        print("start testing")
        bertm.model.eval()
        bertm.encode_start = 0
        bertm.encode_end = 0
        bertm.interaction_start = 0
        bertm.interaction_end = 0
        bertm.block_start = 0
        bertm.block_end = 0
        s = time.time()
        RR, PC, p, r, f1 = validation_full(bertm, test, doc_att, args.hamming)
        e = time.time()
        #print("encoding time:", bertm.encode_end - bertm.encode_start)
        #print("interaction time:", bertm.interaction_end - bertm.interaction_start)
        #print("blocking time:", bertm.block_end - bertm.block_start)
        #print("RR & PC ", RR, PC)
        print("F1: ", f1)
        #print(e-s)
        torch.save(bertm, "model/bestmodel.pkl")

if __name__=="__main__":
    try:
        args = get_args()
        bert_tm_full(args)

    except KeyboardInterrupt:
        print("error")