import os
import unicodedata
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch import optim
import time
import math
import random
from torchtext.data.utils import get_tokenizer
from pyvi.ViTokenizer import tokenize as Vitoken
from collections import Counter
from torchtext.vocab import build_vocab_from_iterator, Vocab
import io
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from typing import Tuple
from torch.utils.data.dataset import random_split
import argparse


parser = argparse.ArgumentParser(description='Process some integers.')
# parser.add_argument("--local_rank", default=0)
# parser.add_argument('--lr', type=float, default=0.0002, help='Learning Rate')
# parser.add_argument('--lamda', type=float, default=1, help='Control regular item size')
args = parser.parse_args()

MAX_LENGTH = 50

# torch.cuda.set_device(7)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalizeString(s):
    # 将 Unicode 字符转换为 ASCII
    Ascii = []
    for c in unicodedata.normalize('NFD', s):
        if unicodedata.category(c) != 'Mn':
            Ascii.append(c)
    # 将ASCII列表转化为字符串，并将所有内容都转换为小写，并修剪大多数标点符号
    s = ''.join(Ascii).lower().strip()
    s = re.sub(r"[^a-z,.!?]+", r" ", s)  # 只保留a-zA-Z.!?并在其后加空格
    return s


def build_vocab_en(filepath, tokenizer):  # specials=['<unk>', '<pad>', '<bos>', '<eos>']
    print('build vocab en...')
    def yield_tokens():
        with io.open(filepath, encoding="utf8") as f:
            for string_ in f:
                yield tokenizer(string_.strip().replace('_', ' '))

    return build_vocab_from_iterator(yield_tokens(), specials=['<unk>', '<pad>', '<bos>', '<eos>'], min_freq=7)


def build_vocab_vi(filepath):
    print('build vocab vi...')

    def yield_tokens():
        with io.open(filepath, encoding="utf8") as f:
            for string_ in f:
                yield Vitoken(string_.strip().replace('_', ' ')).split(' ')

    return build_vocab_from_iterator(yield_tokens(), specials=['<unk>', '<pad>', '<bos>', '<eos>'], min_freq=7)

    # counter = Counter()
    # with io.open(filepath, encoding="utf8") as f:
    #     for string_ in f:
    #         counter.update(Vitoken(string_))
    # return build_vocab_from_iterator(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])


def data_process(filepaths1, filepaths2):
    print('data process...')
    raw_vi_iter = iter(io.open(filepaths1, encoding="utf8"))      # 生成迭代器
    raw_en_iter = iter(io.open(filepaths2, encoding="utf8"))      # <_io.TextIOWrapper name='... your path\\.data\\train.en' mode='r' encoding='utf8'>
    data = []
    for (raw_vi, raw_en) in zip(raw_vi_iter, raw_en_iter):      # raw_de, raw_en就是每一句话
        if len(raw_vi.strip().split(' ')) < MAX_LENGTH and len(raw_en.strip().split(' ')) < MAX_LENGTH:
            vi_tensor_ = torch.tensor([vi_vocab[token] for token in Vitoken(raw_vi)], dtype=torch.long)
            en_tensor_ = torch.tensor([en_vocab[token] for token in en_tokenizer(raw_en)], dtype=torch.long)
            data.append((vi_tensor_, en_tensor_))
    return data


def generate_batch(data_batch):     # BATCH_SIZE = 128
    # print('generate batch...')
    vi_batch, en_batch = [], []
    for (vi_item, en_item) in data_batch:
        vi_batch.append(torch.cat([torch.tensor([BOS_IDX]), vi_item, torch.tensor([EOS_IDX])], dim=0))
        en_batch.append(torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
    # len(vi_batch) = 128
    vi_batch = pad_sequence(vi_batch, padding_value=PAD_IDX)        # <class 'torch.Tensor'>, torch.Size([*, 128])
    en_batch = pad_sequence(en_batch, padding_value=PAD_IDX)
    return vi_batch, en_batch


class Encoder(nn.Module):
    def __init__(self, input_dim: int, emb_dim: int, enc_hid_dim:int, dec_hid_dim: int, dropout: float):    # seq_len, 32, 64, 64, 0.5
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout

        self.embedding = nn.Embedding(input_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True)
        self.fc = nn.Linear(enc_hid_dim * 2, dec_hid_dim)

    def forward(self, src: torch.Tensor) -> Tuple[torch.Tensor]:        # 函数参数中的冒号是参数的类型建议符，告诉程序员希望传入的实参的类型。函数后面跟着的箭头是函数返回值的类型建议符，用来说明该函数返回的值是什么类型
        embedded = self.dropout(self.embedding(src))        # seq_len * 128 * emb_dim
        outputs, hidden = self.rnn(embedded)        # 为什么没有hidden?因为这里直接把一句话输入，而不是一个词一个词输入
        # hidden = torch.tanh(self.fc(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)     # num_layers * num_directions, batch, hidden_size → torch.Size([128, 64*2])
        hidden = self.fc(hidden)      # torch.Size([128, 64])
        hidden = torch.tanh(hidden)     # 双曲正切函数, 激活函数, torch.Size([128, 64])
        return outputs, hidden      # seq_len, batch(128), num_directions(2) * hidden_size(64), torch.Size([128, 64])


class Attention(nn.Module):
    def __init__(self, enc_hid_dim: int, dec_hid_dim: int, attn_dim: int):
        super().__init__()
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.attn_in = (enc_hid_dim * 2) + dec_hid_dim
        self.attn = nn.Linear(self.attn_in, attn_dim)

    def forward(self, decoder_hidden, encoder_outputs) -> torch.Tensor:
        src_len = encoder_outputs.shape[0]
        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((repeated_decoder_hidden, encoder_outputs), dim=2)))
        attention = torch.sum(energy, dim=2)
        return F.softmax(attention, dim=1)


class Decoder(nn.Module):
    def __init__(self, output_dim: int, emb_dim: int, enc_hid_dim: int, dec_hid_dim: int, dropout: int, attention: nn.Module):
        super().__init__()
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.attention = attention

        self.embedding = nn.Embedding(output_dim, emb_dim)
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim)
        self.out = nn.Linear(self.attention.attn_in + emb_dim, output_dim)

    def _weighted_encoder_rep(self, decoder_hidden, encoder_outputs) -> torch.Tensor:
        a = self.attention(decoder_hidden, encoder_outputs)
        a = a.unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        weighted_encoder_rep = torch.bmm(a, encoder_outputs)
        weighted_encoder_rep = weighted_encoder_rep.permute(1, 0, 2)
        return weighted_encoder_rep

    def forward(self, input: torch.Tensor, decoder_hidden: torch.Tensor, encoder_outputs: torch.Tensor) -> Tuple[torch.Tensor]:
        # input.size() = torch.Size([128])
        input = input.unsqueeze(0)      # 1 * 128
        embedded = self.dropout(self.embedding(input))      # 1* 128 * 32
        weighted_encoder_rep = self._weighted_encoder_rep(decoder_hidden, encoder_outputs)      # torch.Size([128, 64]), torch.Size([len, 128, 128]), 1*128*128
        rnn_input = torch.cat((embedded, weighted_encoder_rep), dim=2)      # 1 *128 * (32+128) torch.Size([1, 128, 160])
        output, decoder_hidden = self.rnn(rnn_input, decoder_hidden.unsqueeze(0))       # 1*128*160, 1*128*64 ,1*128*64, 1*128*64

        embedded = embedded.squeeze(0)  # torch.Size([128, 32])
        output = output.squeeze(0)      # 128*64
        weighted_encoder_rep = weighted_encoder_rep.squeeze(0)      # 128*128
        output = self.out(torch.cat((output, weighted_encoder_rep, embedded), dim=1))       # 128*len(en)

        return output, decoder_hidden.squeeze(0)        # 128*len(en), 128*64


class Seq2Seq(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module, device: torch.device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src: torch.Tensor, trg: torch.Tensor, teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        batch_size = src.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        encoder_outputs, hidden = self.encoder(src)

        output = trg[0,:]   # first input to the decoder is the <sos> token
        # output = tensor([2, 2, 2, 2, 2 ...(128个)
        for t in range(1, max_len):
            output, hidden = self.decoder(output, hidden, encoder_outputs)
            outputs[t] = output     # torch.Size([128, len(en)(6416)])
            teacher_force = random.random() < teacher_forcing_ratio     # teacher_force True(From Reference) or False(From Model)?
            top1 = output.max(1)[1]
            output = (trg[t] if teacher_force else top1)

        return outputs


def init_weights(m: nn.Module):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def train(model: nn.Module, iterator: torch.utils.data.DataLoader, optimizer: optim.Optimizer, criterion: nn.Module, clip: float):
    print('training...')
    model.train()       # 模式选择
    epoch_loss = 0
    for _, (src, trg) in enumerate(iterator):
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg)        # max_len, batch_size, trg_vocab_size
        output = output[1:].view(-1, output.shape[-1])      # torch.Size([(max_len-1)*128, 6416])
        trg = trg[1:].view(-1)      # # torch.Size([(max_len-1)*128])
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        if (_ + 1) % 10 == 0:
            print('\r' + str(_ + 1), end='')
    print()
    return epoch_loss / len(iterator)


def evaluate(model: nn.Module, iterator: torch.utils.data.DataLoader, criterion: nn.Module):
    print('evaluate...')
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for _, (src, trg) in enumerate(iterator):
            src, trg = src.to(device), trg.to(device)
            output = model(src, trg, 0) #turn off teacher forcing
            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
            if (_ + 1) % 10 == 0:
                print('\r' + str(_ + 1), end='')
        print()
    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def test(sentence):
    vi_tensor = torch.tensor([vi_vocab[token] for token in Vitoken(sentence)], dtype=torch.long)
    vi_tensor = torch.cat([torch.tensor([BOS_IDX]), vi_tensor, torch.tensor([EOS_IDX])], dim=0)
    vi_tensor = vi_tensor.unsqueeze(1).to(device)
    output = model(vi_tensor, torch.zeros(50, 1).long().to(device), 0)
    output = output[1:].view(-1, output.shape[-1])  # torch.Size([49, 6416])
    result = []
    for i in range(output.size()[0]):
        index = output[i].data.topk(1)[1].item()
        if en_vocab.get_itos()[index] == '<eos>':
            break
        result.append(en_vocab.get_itos()[index])  # 1722, apple
    print(' '.join(result))


if __name__ == '__main__':
    root = 'iwslt15_en_vt'
    print('Reading lines...')
    train_X_path = root + '/train.vi'
    train_Y_path = root + '/train.en'
    en_tokenizer = get_tokenizer("basic_english")
    vi_vocab = build_vocab_vi(train_X_path)
    en_vocab = build_vocab_en(train_Y_path, en_tokenizer)

    vi_vocab.set_default_index(0)
    en_vocab.set_default_index(0)
    PAD_IDX = vi_vocab['<pad>']  # 1
    BOS_IDX = vi_vocab['<bos>']  # 2
    EOS_IDX = vi_vocab['<eos>']  # 3
    print(PAD_IDX, BOS_IDX, EOS_IDX)

    train_data = data_process(train_X_path, train_Y_path)
    print('len of data:' + str(len(train_data)))
    BATCH_SIZE = 128

    n_train = int(len(train_data) * 0.8)
    split_train, split_valid = random_split(dataset=train_data, lengths=[n_train, len(train_data) - n_train])

    train_iter = DataLoader(split_train, batch_size=BATCH_SIZE, collate_fn=generate_batch, shuffle=True)
    valid_iter = DataLoader(split_valid, batch_size=BATCH_SIZE, collate_fn=generate_batch, shuffle=True)

    test_X_path = root + '/tst2012.vi'
    test_Y_path = root + '/tst2012.en'
    test_data = data_process(test_X_path, test_Y_path)
    print('len of test data:' + str(len(test_data)))
    test_iter = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=generate_batch)

    INPUT_DIM = len(vi_vocab)
    OUTPUT_DIM = len(en_vocab)
    print(INPUT_DIM, OUTPUT_DIM)

    ENC_EMB_DIM = 64
    DEC_EMB_DIM = 64
    ENC_HID_DIM = 128
    DEC_HID_DIM = 128
    ATTN_DIM = 16
    ENC_DROPOUT = 0.2
    DEC_DROPOUT = 0.2
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    attn = Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)
    model = Seq2Seq(enc, dec, device).to(device)

    # model.apply(init_weights)
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index=en_vocab['<pad>'])

    model.load_state_dict(torch.load('50/model_Translate_10.pth'))

    N_EPOCHS = 10
    CLIP = 1
    best_valid_loss = float('inf')
    for epoch in range(N_EPOCHS):       # 最大迭代次数
        start_time = time.time()
        train_loss = train(model, train_iter, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, valid_iter, criterion)
        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    test_loss = evaluate(model, test_iter, criterion)

    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

    torch.save(model.state_dict(), '50/model_Translate_20.pth')


    # model.load_state_dict(torch.load('50/model_Translate_10.pth'))
    test('Câu chuyện này chưa kết thúc .')
    test('Ông rút lui vào yên lặng .')
    test('Ông qua đời , bị lịch sử quật ngã .')
    test('Ông là ông của tôi .')
    test('Tôi chưa bao giờ gặp ông ngoài đời .')

