import torch

class Hamming_dict():
    def __init__(self, d):
        super(Hamming_dict, self).__init__()
        self.buckets = {}
        self.key = []
        self.multi = {}
        self.d = d

    def build_bucket(self, hc):
        for tc in hc:
            c = [int(x) for x in tc.tolist()]
            sc = ''.join(map(str, c))
            if sc not in self.buckets:
                self.buckets[sc] = []
                self.multi[sc] = 1
                self.key.append(tc.unsqueeze(dim=0))
            else:
                self.multi[sc] = self.multi[sc] + 1

    def _get_keys(self, t_key, c):
        Hamming_d = torch.sum((t_key - c).abs(), dim=1)
        indices = (Hamming_d <= self.d).nonzero().squeeze(dim=1)
        if indices is not None:
            keys = torch.index_select(t_key, 0, indices)
        else:
            keys = None
        return keys

    def insert_bucket(self, hc):
        t_key = torch.cat(self.key, dim=0)
        for tc in hc:
            keys = self._get_keys(t_key, tc.unsqueeze(0))
            if keys is not None:
                for key in keys:
                    c = [int(x) for x in key.tolist()]
                    sc = ''.join(map(str, c))
                    self.buckets[sc].append(sc)

    def compute_B(self):
        count = 0
        for key in self.buckets.keys():
            count += len(self.buckets[key]) * self.multi[key]
        return count

