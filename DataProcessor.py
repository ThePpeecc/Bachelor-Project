import pickle
import torch
import numpy as np

class DataInstance():
    # Takes sets itself up like one document, it contains ids, token types and masks
    def __init__(self, ids, types, mask, text = '', label = -1):
        self.ids = torch.tensor(ids)
        self.types = torch.tensor(types)
        self.mask = torch.tensor(mask)
        self.text = text
        self.label = label
        self.preds = torch.tensor([])

    def predict(self, model):
        with torch.no_grad():
            _, x = model(self.ids.unsqueeze(0), self.mask.unsqueeze(0), self.types.unsqueeze(0))
            self.preds = x
            return x

class DataProcesser():
    def flat(self, d): return [x for y in d for x in y]

    def __init__(self, data = None, labels = None, tokenizer = None):
        self.onCuda = False
        if (data == None and labels == None):
            return

        flattend = self.flat(data)
        encoded = None
        if tokenizer != None:
            encoded = tokenizer.batch_encode_plus(flattend, pad_to_max_length=True)

        instanceSet = []
        for i in range(len(flattend)):
            
            ids = []
            types = []
            mask = []

            if encoded != None:
                ids = encoded['input_ids'][i]
                types = encoded['token_type_ids'][i]
                mask = encoded['attention_mask'][i]
            
            text = flattend[i]
            label = labels[i]

            instanceSet.append(DataInstance(ids, types, mask, text, label))

        self.instances = instanceSet

    def ids(self):
        if self.onCuda:
            return self.cids
        return torch.tensor([x.ids.numpy() for x in self.instances])

    def types(self):
        if self.onCuda:
            return self.ctypes
        return torch.tensor([x.types.numpy() for x in self.instances])

    def masks(self):
        if self.onCuda:
            return self.cmasks
        return torch.tensor([x.mask.numpy() for x in self.instances])

    def preds(self):
        if self.onCuda:
            return self.cpreds
        return torch.tensor([x.preds.numpy() for x in self.instances])

    def labels(self):
        if self.onCuda:
            return self.clabels
        return torch.tensor([x.label.numpy() for x in self.instances])

    def predictWith(self, model):
        print("Running data on model")
        for i, inst in enumerate(self.instances):
            inst.predict(model)
            print("Instance: {} saved!".format(i))

        print("Done")

    def cuda(self):
        self.cids = self.ids().cuda()
        self.ctypes = self.types().cuda()
        self.cmasks = self.masks().cuda()
        self.clabels = self.labels().cuda()
        self.cpreds = self.preds().cuda()
        self.onCuda = True

    def cpu(self):
        self.onCuda = False


    def FromInstances(insts):
        new = DataProcesser()
        new.instances = insts
        return new

    def save(self, to, hide = True):
        if hide:
            for inst in self.instances:
                inst.text = None # We remove any text
                #inst.ids = None # Any potential open ids for decoding
                #inst.mask = None # Mask is not nesssary anymore
                #inst.types = None # types is not nesssary anymore


        file = open(to, 'wb')
        return pickle.dump(self, file)

    def load(from_file):
        file = open(from_file, 'rb')
        new = pickle.load(file)
        return new


