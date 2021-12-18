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
from torchtext.vocab import build_vocab_from_iterator
import io
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from typing import Tuple
from torch.utils.data.dataset import random_split
import argparse


parser = argparse.ArgumentParser(description='Process some integers.')
# parser.add_argument('--lr', type=float, default=0.0002, help='Learning Rate')
args = parser.parse_args()

MAX_LENGTH = 10

# torch.cuda.set_device(7)
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_vocab(filepath_list, tokenizer):
    print('build vocab ...')
    def yield_tokens():
        for filepath in filepath_list:
            with io.open(filepath, 'r', encoding="utf8") as f:
                for string_ in f:
                    yield tokenizer(string_.strip())
    return build_vocab_from_iterator(yield_tokens(), specials=['<unk>', '<pad>', '<bos>', '<eos>'], min_freq=5)


# eng_prefixes = (
#     "i am ",
#     "he is",
#     "she is",
#     "you are",
#     "we are",
#     "they are"
# )


def data_process(filepaths1_list, filepaths2_list):
    print('data process...')
    if len(filepaths1_list) != len(filepaths2_list):
        print('Wrong!')
    data = []
    for _ in range(len(filepaths1_list)):
        raw_en_iter = iter(io.open(filepaths1_list[_], encoding="utf8"))
        raw_de_iter = iter(io.open(filepaths2_list[_], encoding="utf8"))      # 生成迭代器
        for (raw_en, raw_de) in zip(raw_en_iter, raw_de_iter):      # raw_de, raw_en就是每一句话
            if len(raw_de.strip().split(' ')) < MAX_LENGTH and len(raw_en.strip().split(' ')) < MAX_LENGTH:
                    # and raw_en.lower().startswith(eng_prefixes):

                en_list = [en_vocab[token] for token in en_tokenizer(raw_en.strip().lower())]
                de_list = [de_vocab[token] for token in de_tokenizer(raw_de.strip().lower())]

                # if all(en_list) and all(de_list):
                en_tensor_ = torch.tensor(en_list, dtype=torch.long)
                de_tensor_ = torch.tensor(de_list, dtype=torch.long)
                data.append((en_tensor_, de_tensor_))
                # if len(data) == 1000:     # for debug
                #     return data
    return data


def generate_batch(data_batch):     # BATCH_SIZE = 128
    # print('generate batch...')
    de_batch, en_batch = [], []
    for (en_item, de_item) in data_batch:
        en_batch.append(torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
        de_batch.append(torch.cat([torch.tensor([BOS_IDX]), de_item, torch.tensor([EOS_IDX])], dim=0))
    # len(de_batch) = 128
    en_batch = pad_sequence(en_batch, padding_value=PAD_IDX)
    de_batch = pad_sequence(de_batch, padding_value=PAD_IDX)        # <class 'torch.Tensor'>, torch.Size([*, 128])
    return en_batch, de_batch


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
        self.rnn = nn.GRU(emb_dim, enc_hid_dim, bidirectional=True, num_layers=2)
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
        # 20211215 en to de 修改
        repeated_decoder_hidden = decoder_hidden.unsqueeze(1).repeat(1, src_len, 1)     # .unsqueeze(1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.cat((repeated_decoder_hidden, encoder_outputs), dim=2)
        energy = torch.tanh(self.attn(energy))
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
        self.rnn = nn.GRU((enc_hid_dim * 2) + emb_dim, dec_hid_dim, bidirectional=False)
        # 20211215 en to de 修改
        self.out = nn.Linear(self.attention.attn_in + emb_dim, output_dim)      #  + ENC_HID_DIM

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
        # 20211215 en to de 修改
        output, decoder_hidden = self.rnn(rnn_input, decoder_hidden.unsqueeze(0).contiguous())       # 1*128*160, 1*128*64 ,1*128*64, 1*128*64, unsqueeze(0)repeat(2,1,1)

        embedded = embedded.squeeze(0)  # torch.Size([128, 32])
        output = output.squeeze(0)      # 128*64
        weighted_encoder_rep = weighted_encoder_rep.squeeze(0)      # 128*128
        output = torch.cat((output, weighted_encoder_rep, embedded), dim=1)
        output = self.out(output)
        # output = self.out(torch.cat((output, weighted_encoder_rep, embedded), dim=1))       # 128*len(en)

        # 20211215 en to de 修改 .squeeze(0)
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
    en_tensor = torch.tensor([en_vocab[token] for token in en_tokenizer(sentence.strip().lower())], dtype=torch.long)
    en_tensor = torch.cat([torch.tensor([BOS_IDX]), en_tensor, torch.tensor([EOS_IDX])], dim=0)
    en_tensor = en_tensor.unsqueeze(1).to(device)
    output = model(en_tensor, torch.zeros(50, 1).long().to(device), 0)
    output = output[1:].view(-1, output.shape[-1])  # torch.Size([49, 6416])
    result = []
    for i in range(output.size()[0]):
        index = output[i].data.topk(1)[1].item()
        if en_vocab.get_itos()[index] == '<eos>':
            break
        result.append(en_vocab.get_itos()[index])  # 1722, apple
    print(' '.join(result))


if __name__ == '__main__':
    root = 'wmt16_de_en'
    print('Reading lines...')
    train_X_path_1 = root + '/europarl-v7.de-en.en'
    train_Y_path_1 = root + '/europarl-v7.de-en.de'
    train_X_path_2 = root + '/commoncrawl.de-en.en'
    train_Y_path_2 = root + '/commoncrawl.de-en.de'
    debug_X_path_1 = root + '/newssyscomb2009.en'
    debug_Y_path_1 = root + '/newssyscomb2009.de'
    debug_X_path_2 = root + '/news-test2008.en'
    debug_Y_path_2 = root + '/news-test2008.de'
    debug_X_path_3 = '.data/multi30k/train.en'
    debug_Y_path_3 = '.data/multi30k/train.de'
    debug_X_path_4 = '.data/multi30k/val.en'
    debug_Y_path_4 = '.data/multi30k/val.de'

    # en_tokenizer = get_tokenizer("basic_english")
    en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
    de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')

    en_vocab = build_vocab([debug_X_path_1, debug_X_path_2], en_tokenizer)      # train_X_path_1, train_X_path_2,
    de_vocab = build_vocab([debug_Y_path_1, debug_Y_path_2], de_tokenizer)      # train_Y_path_1, train_Y_path_2
    # en_vocab = build_vocab([train_X_path_1], en_tokenizer)  # train_X_path_1, train_X_path_2,
    # de_vocab = build_vocab([train_Y_path_1], de_tokenizer)  # train_Y_path_1, train_Y_path_2

    en_vocab.set_default_index(0)
    de_vocab.set_default_index(0)

    PAD_IDX = en_vocab['<pad>']  # 1
    BOS_IDX = en_vocab['<bos>']  # 2
    EOS_IDX = en_vocab['<eos>']  # 3
    print(PAD_IDX, BOS_IDX, EOS_IDX)

    # all_data = data_process([train_X_path_1, train_X_path_2], [train_Y_path_1, train_Y_path_2])
    all_data = data_process([debug_X_path_1, debug_X_path_2], [debug_Y_path_1, debug_Y_path_2])
    # all_data = data_process([train_X_path_1], [train_Y_path_1])     # min_freq=30

    print('len of data:' + str(len(all_data)))
    BATCH_SIZE = 96

    n_train = int(len(all_data) * 0.8)
    n_valid = int(len(all_data) * 0.15)
    split_train, split_valid, split_test = random_split(dataset=all_data, lengths=[n_train, n_valid, len(all_data) - n_train - n_valid])

    train_iter = DataLoader(split_train, batch_size=BATCH_SIZE, collate_fn=generate_batch, shuffle=True)
    valid_iter = DataLoader(split_valid, batch_size=BATCH_SIZE, collate_fn=generate_batch, shuffle=True)
    test_iter = DataLoader(split_test, batch_size=BATCH_SIZE, collate_fn=generate_batch)

    INPUT_DIM = len(en_vocab)
    OUTPUT_DIM = len(de_vocab)
    print(INPUT_DIM, OUTPUT_DIM)

    ENC_EMB_DIM = 128
    DEC_EMB_DIM = 128
    ENC_HID_DIM = 256
    DEC_HID_DIM = 256
    ATTN_DIM = 32
    ENC_DROPOUT = 0.2
    DEC_DROPOUT = 0.2
    enc = Encoder(INPUT_DIM, ENC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, ENC_DROPOUT)
    attn = Attention(ENC_HID_DIM, DEC_HID_DIM, ATTN_DIM)
    dec = Decoder(OUTPUT_DIM, DEC_EMB_DIM, ENC_HID_DIM, DEC_HID_DIM, DEC_DROPOUT, attn)
    model = Seq2Seq(enc, dec, device).to(device)

    model.apply(init_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(ignore_index=en_vocab['<pad>'])

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
        torch.save(model.state_dict(), 'worked/model_{}.pt'.format(epoch + 1))

    test_loss = evaluate(model, test_iter, criterion)

    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')


    for i, item in enumerate(split_test):
        sentence1 = []
        for _ in item[0]:
            sentence1.append(en_vocab.get_itos()[_.item()])
        sentence1 = ' '.join(sentence1)
        sentence2 = []
        for _ in item[0]:
            sentence2.append(en_vocab.get_itos()[_.item()])
        sentence2 = ' '.join(sentence2)
        print(sentence1, sentence2)
        print(test(sentence1))
        if i == 5:
            break

    # model.load_state_dict(torch.load('50/model_Translate_10.pth'))
    test('i am looking forward to your reply.')
    test('i love you.')
    test('I love big data analytics.')
    test('Let me introduce myself.')
    test('What day is it today?')




