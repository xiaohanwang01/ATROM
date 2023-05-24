import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from data_process import DataGenerator
from sklearn import metrics
from sklearn.metrics import f1_score as F1
from sklearn.metrics import roc_auc_score as auc


#args
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_dim = 256
map_size = (51,158)
input_size = 51*158+1
num_classes = 3
output_size = 51*158+3
token_dim = 128
hidden_size = 256
z_dim=128
layer = 1
batch_size = 32
lr = 0.001


def kl_div(p_mean, p_log_var, t_mean, t_log_var):
    kl = - 0.5 * (p_log_var - t_log_var + 1 - torch.exp(p_log_var) / torch.exp(t_log_var) - (p_mean - t_mean).pow(2) / torch.exp(t_log_var)).sum(-1)
    return kl


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embbed_dim, num_layers, embedding):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.embbed_dim = embbed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.linear = nn.Linear(embedding_dim, token_dim)
        self.embedding = nn.Embedding.from_pretrained(embedding, padding_idx=0)
        self.gru = nn.GRU(self.embbed_dim, self.hidden_dim, num_layers=self.num_layers, batch_first=True)

    def forward(self, src, lengths):
        embedded = self.linear(self.embedding(src))
        pack = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        output, _ = self.gru(pack)
        output , _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        hidden = output.mean(1).unsqueeze(0)
        return hidden

class Decoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, embbed_dim, num_layers):
        super(Decoder, self).__init__()
        
        self.embbed_dim = embbed_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        self.embedding = nn.Embedding(output_dim, self.embbed_dim, padding_idx=0)
        self.gru = nn.GRU(self.embbed_dim+self.hidden_dim+num_classes, self.hidden_dim, num_layers=self.num_layers, batch_first=True)
        self.out = nn.Linear(self.hidden_dim, output_dim)  
        self.fc = nn.Linear(num_classes*hidden_size, hidden_size)
        
    def forward(self, input, encode_hidden, decode_hidden, lengths, y_oh):
        embedded = self.embedding(input)
        labels = y_oh.unsqueeze(1).repeat(1,embedded.shape[1],1)
        encode_hidden = encode_hidden.transpose(0,1).repeat(1,embedded.shape[1],1)
        embedded = torch.cat((embedded, encode_hidden, labels), 2)
        pack = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        decode_hidden = self.fc(decode_hidden.view(decode_hidden.shape[0], -1)).unsqueeze(0)
        output, _ = self.gru(pack, decode_hidden)
        output , _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        prediction = self.out(output)
        return prediction


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, t_mu_shift, t_var_scale):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.num_classes = num_classes
        self.z_dim = z_dim

        self.fc1 = nn.Linear(self.z_dim, num_classes)
        self.fc2 = nn.Linear(hidden_size, self.num_classes * self.z_dim) 
        self.fc3 = nn.Linear(hidden_size, self.num_classes * self.z_dim)
        self.fc4 = nn.Linear(self.z_dim, hidden_size)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.t_mean = nn.Embedding(self.num_classes, self.num_classes * self.z_dim)
        self.t_log_var = nn.Embedding(self.num_classes, self.num_classes * self.z_dim)
        self.t_mean, self.t_log_var = self._init_targets(t_mu_shift, t_var_scale)
        
        
    def _init_targets(self, t_mu_shift, t_var_scale):
        t_mean_init = 1.0 * F.one_hot(torch.arange(self.num_classes), self.num_classes)
        t_mean_init = t_mean_init.unsqueeze(-1).repeat(1, 1, self.z_dim).view(self.num_classes, -1)
        t_mean_init = t_mean_init * t_mu_shift
        t_mean = nn.Embedding.from_pretrained(t_mean_init, freeze=False)
        t_log_var_init = torch.ones(self.num_classes, self.num_classes * self.z_dim)
        t_log_var_init = t_log_var_init * t_var_scale
        t_log_var = nn.Embedding.from_pretrained(t_log_var_init, freeze=False)
        return t_mean, t_log_var
    
    def cross_kl_div(self, z_mu, z_log_var, detach_inputs=False, detach_targets=False):
        """
        Compute kl divergence between the variation capsule distribution defined by z_mu and z_var with all the 
        targets.
        
        Args:
            z_mu: tensor of shape [bs, num_classes, z_dim].
            z_var: tensor of shape [bs, num_classes, z_dim].
            detach_inputs: bool.
            detach_targets: bool.
            
        Ouput:
            kl: tensor of shape [bs, num_classes]
        """
        B, num_classes, _ = z_mu.shape
        kl = []
        t_idxs = torch.arange(num_classes).unsqueeze(0).repeat(B, 1).to(z_mu.device)
        t_means = self.t_mean(t_idxs) 
        t_log_vars = self.t_log_var(t_idxs) 
        
        if detach_inputs:
            z_mu = z_mu.detach()
            z_log_var = z_log_var.detach()
            
        if detach_targets:
            t_means = t_means.detach()
            t_log_vars = t_log_vars.detach()
            
        for t_mean_i, t_log_var_i in zip(t_means.permute(1, 0, 2), t_log_vars.permute(1, 0, 2)):
            kl_i = kl_div(torch.flatten(z_mu, 1), torch.flatten(z_log_var, 1), t_mean_i, t_log_var_i) 
            kl += [kl_i]
        kl = torch.stack(kl, 1) 
        return kl

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self,batch_encode_input, batch_decode_input, batch_lengths, y_oh=None, train=True):
        encode_hidden = self.encoder(batch_encode_input, batch_lengths)
        mu = self.fc2(encode_hidden.squeeze(0)).view(encode_hidden.shape[1], self.num_classes, z_dim)
        log_var = self.fc3(encode_hidden.squeeze(0)).view(encode_hidden.shape[1], self.num_classes, z_dim)
        z = self.reparameterize(mu, log_var)
        decode_hidden = self.relu1(self.fc4(z))

        kl = self.cross_kl_div(mu, log_var, detach_targets=True)
        logits = - torch.log(kl)

        batch_lengths += torch.tensor(1) 
        
        if(train):
            batch_decode_output = self.decoder(batch_decode_input, encode_hidden, decode_hidden, batch_lengths, y_oh)
        else:
            batch_decode_output = self.decoder(batch_decode_input, encode_hidden, decode_hidden, batch_lengths, logits)

        out = {
            'encode_hidden': encode_hidden,
            'rec': batch_decode_output,
            'logits': logits,
            'z': z,
            'z_mu': mu,
            'z_log_var': log_var,
            'kl': kl
        }
        return out

def train(model, data, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for batch_encode_input, batch_decode_input, batch_decode_output, batch_seq_length, labels in data.iterate_data(data_type='train'):   
        batch_encode_input, batch_decode_input, batch_decode_output, batch_seq_length, labels = batch_encode_input.to(device), batch_decode_input.to(device), batch_decode_output.to(device), batch_seq_length.to(device), labels.to(device)
        
        y_oh = F.one_hot(labels, num_classes)
        preds = model(batch_encode_input, batch_decode_input, batch_seq_length, y_oh)
        t_mu, t_log_var = model.t_mean(labels), model.t_log_var(labels)
        t_mu = t_mu.view(preds['z_mu'].shape)
        t_log_var = t_log_var.view(preds['z_log_var'].shape)

        loss_kl = kl_div(preds['z_mu'], preds['z_log_var'], t_mu.detach(), t_log_var.detach()).mean(-1).mean(0)
        loss_contr = model.cross_kl_div(preds['z_mu'], preds['z_log_var'], detach_inputs=True)
        loss_contr = torch.where(y_oh < 1, loss_contr, torch.zeros_like(loss_contr))
        loss_contr = F.relu(10 - loss_contr)
        loss_contr = (loss_contr / (num_classes - 1)).sum(-1).mean(0)
        loss_rec = criterion(preds['rec'].transpose(1,2), batch_decode_output).mean()
        loss = loss_kl + loss_rec + loss_contr
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / data.train_traj_num * batch_size


def get_threshold(model, data):
    values = []
    with torch.no_grad():
        for batch_encode_input, batch_decode_input, batch_decode_output, batch_seq_length, labels in data.iterate_data(
                data_type='train'):
            batch_encode_input, batch_decode_input, batch_decode_output, batch_seq_length, labels = \
                batch_encode_input.to(device), batch_decode_input.to(device), batch_decode_output.to(
                    device), batch_seq_length.to(device), labels.to(device)

            preds = model(batch_encode_input, batch_decode_input, batch_seq_length, train=False)
            value, _ = torch.max(F.softmax(preds['logits'], -1), -1)
            values.append(list(value.to('cpu')))
    values = np.array(values).reshape(-1)
    values = np.sort(values)
    threshold = values[int(np.floor(len(values)*0.1))]

    return threshold


def _test(model, data, threshold):
    target = []
    y_pred = []
    binary = []
    binary_pred = []
    with torch.no_grad():
        for batch_encode_input, batch_decode_input, batch_decode_output, batch_seq_length, labels in data.iterate_data(data_type='test'):
            batch_encode_input, batch_decode_input, batch_decode_output, batch_seq_length, labels = \
                batch_encode_input.to(device), batch_decode_input.to(device), batch_decode_output.to(device), batch_seq_length.to(device), labels.to(device)
           
            preds = model(batch_encode_input, batch_decode_input, batch_seq_length, train=False)

            target.append(labels.to('cpu').numpy())
            
            values, indices = torch.max(F.softmax(preds['logits'], -1), -1)
            opens = (torch.zeros_like(values, dtype=torch.long) + 3).to(device)
            ones = (torch.zeros_like(values, dtype=torch.long) + 1).to(device)
            zeros = torch.zeros_like(values, dtype=torch.long).to(device)
            indices = torch.LongTensor(indices.to('cpu')).to(device)
            result = torch.where(values > threshold, indices, opens)
            binary_result = torch.where(labels == 3, zeros, ones)
            y_pred.append(result.to('cpu').numpy())
            binary.append(binary_result.to('cpu').numpy())
            binary_pred.append(values.to('cpu').numpy())

    return target, y_pred, binary, binary_pred


if __name__ == "__main__":
    embedding = torch.FloatTensor(np.load('./embedding/embedding1.npy'))
    encoder = Encoder(input_size, hidden_size, token_dim, layer, embedding)
    decoder = Decoder(output_size, hidden_size, token_dim, layer)
    model = Seq2Seq(encoder, decoder, device, t_mu_shift=1.0, t_var_scale=1.0).to(device)
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.SGD(model.parameters(), lr = lr, momentum=0.9)
    data = DataGenerator()

    data.load_outliers('train_porto_outliers.pkl', 'train')
    for epoch in range(1, 11):
        print('epoch:{} \tloss:{}'.format(epoch, train(model, data, optimizer, criterion)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.0001
    for epoch in range(11, 51):
        print('epoch:{} \tloss:{}'.format(epoch, train(model, data, optimizer, criterion)))

    threshold = get_threshold(model, data)

    data.load_outliers('test_porto_outliers.pkl', 'test')
    target, y_pred, binary, binary_pred = _test(model, data, threshold)
    target = np.array(target).reshape(-1)
    y_pred = np.array(y_pred).reshape(-1)
    binary = np.array(binary).reshape(-1)
    binary_pred = np.array(binary_pred).reshape(-1)
    print('F1-score:{}'.format(F1(target,y_pred,average='macro')))
    print('AUROC:{}'.format(auc(binary,binary_pred,average='macro')))
    precision, recall, _ = metrics.precision_recall_curve(binary,binary_pred)
    area = metrics.auc(recall, precision)
    print('PR-AUC:{}'.format(area))