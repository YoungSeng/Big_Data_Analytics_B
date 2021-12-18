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
import os
# from bleu import file_bleu
# from bleu import list_bleu

MAX_LENGTH = 50     # 3198:不过滤最大长度，单独的一个全局变量

device = torch.device("cuda:2")
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# def get_moses_multi_bleu(hypotheses, references, lowercase=False):
#     import tempfile
#     import numpy as np
#     import subprocess
#     from torchnlp._third_party.lazy_loader import LazyLoader
#     import logging
#     logger = logging.getLogger(__name__)
#
#     six = LazyLoader('six', globals(), 'six')
#
#     """Get the BLEU score using the moses `multi-bleu.perl` script.
#
#     **Script:**
#     https://raw.githubusercontent.com/moses-smt/mosesdecoder/master/scripts/generic/multi-bleu.perl
#
#     Args:
#       hypotheses (list of str): List of predicted values
#       references (list of str): List of target values
#       lowercase (bool): If true, pass the "-lc" flag to the `multi-bleu.perl` script
#
#     Returns:
#       (:class:`np.float32`) The BLEU score as a float32 value.
#     """
#
#     if isinstance(hypotheses, list):
#         hypotheses = np.array(hypotheses)
#     if isinstance(references, list):
#         references = np.array(references)
#
#     if np.size(hypotheses) == 0:
#         return np.float32(0.0)
#
#     # Get MOSES multi-bleu script
#     try:
#         multi_bleu_path, _ = six.moves.urllib.request.urlretrieve(
#             "https://raw.githubusercontent.com/moses-smt/mosesdecoder/"
#             "master/scripts/generic/multi-bleu.perl")
#         os.chmod(multi_bleu_path, 0o755)
#     except:
#         logger.warning("Unable to fetch multi-bleu.perl script")
#         return None
#
#     # Dump hypotheses and references to tempfiles
#     hypothesis_file = tempfile.NamedTemporaryFile()
#     hypothesis_file.write("\n".join(hypotheses).encode("utf-8"))
#     hypothesis_file.write(b"\n")
#     hypothesis_file.flush()
#     reference_file = tempfile.NamedTemporaryFile()
#     reference_file.write("\n".join(references).encode("utf-8"))
#     reference_file.write(b"\n")
#     reference_file.flush()
#
#     # Calculate BLEU using multi-bleu script
#     with open(hypothesis_file.name, "r") as read_pred:
#         bleu_cmd = [multi_bleu_path]
#         if lowercase:
#             bleu_cmd += ["-lc"]
#         bleu_cmd += [reference_file.name]
#         try:
#             bleu_out = subprocess.check_output(bleu_cmd, stdin=read_pred, stderr=subprocess.STDOUT)
#             bleu_out = bleu_out.decode("utf-8")
#             bleu_score = re.search(r"BLEU = (.+?),", bleu_out).group(1)
#             bleu_score = float(bleu_score)
#             bleu_score = np.float32(bleu_score)
#         except subprocess.CalledProcessError as error:
#             if error.output is not None:
#                 logger.warning("multi-bleu.perl script returned non-zero exit code")
#                 logger.warning(error.output)
#             bleu_score = None
#
#     # Close temp files
#     hypothesis_file.close()
#     reference_file.close()
#
#     return bleu_score


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):      #用于在给定当前时间和进度％的情况下打印经过的时间和估计的剩余时间
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '运行时间 %s (估计的剩余时间 %s)' % (asMinutes(s), asMinutes(rs))


def normalizeString(s):
    # 将 Unicode 字符转换为 ASCII
    Ascii = []
    for c in unicodedata.normalize('NFD', s):
        if unicodedata.category(c) != 'Mn':
            Ascii.append(c)
    # 将ASCII列表转化为字符串，并将所有内容都转换为小写，并修剪大多数标点符号
    s = ''.join(Ascii).lower().strip()
    s = re.sub(r"[^a-z,.!?]+", r" ", s)       # 只保留a-zA-Z.!?并在其后加空格
    return s


UNK_token = 0       # unknow word
SOS_token = 1       # Start of sentence
EOS_token = 2       # End of sentence
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "UNK", 1: "SOS", 2: "EOS"}
        self.n_words = 3

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.word2count[word] = 1
            self.n_words += 1
        else:
            self.word2count[word] += 1


def normalizeString_(s):
    # 将 Unicode 字符转换为 ASCII
    Ascii = []
    for c in unicodedata.normalize('NFD', s):
        if unicodedata.category(c) != 'Mn':
            Ascii.append(c)
    # 将ASCII列表转化为字符串，并将所有内容都转换为小写，并修剪大多数标点符号
    # s = ''.join(Ascii).lower().strip()
    s = re.sub(r"[$#@%^&*()/_\-+|\\'\"0-9 ]+", r" ", s)       # 只保留a-zA-Z.!?并在其后加空格
    return s


def readLangs(root, lang1='en', lang2='vi', reverse=False):     # 默认英语→其他语言，可以用reverse反转
    print('Reading lines...')
    # global MAX_LENGTH
    train_X_path = root + '/train.en'
    train_Y_path = root + '/train.vi'
    pairs = []
    with open(train_X_path, 'r', encoding='utf-8') as file_X:
        with open(train_Y_path, 'r', encoding='utf-8') as file_Y:
            lines_X = file_X.read().strip().replace('&amp; amp ; quot ;', '"')\
                .replace('&apos;', "'").replace('&quot;', '"')\
                .replace('&amp;', '&').replace('&#91;', '[')\
                .replace('&#93;', ']').replace('& amp ;', '&').split('\n')
            lines_Y = file_Y.read().strip().replace('&amp; amp ; quot ;', '"')\
                .replace('&apos;', "'").replace('&quot;', '"')\
                .replace('&amp;', '&').replace('&#91;', '[')\
                .replace('&#93;', ']').replace('& amp ;', '&').split('\n')
            # MAX_LENGTH = len(max(max(lines_X, key=len), max(lines_Y, key=len)))
            # print('MAX_LENGTH:' + str(MAX_LENGTH))
            for (line, line_) in zip(lines_X, lines_Y):
                if line != '':
                    pairs.append([normalizeString(line), normalizeString_(line_)])
        # print('Length of pairs:' + str(len(pairs)))
        if reverse:
            pairs = [list(reversed(i)) for i in pairs]
            input_lang = Lang(lang2)  # 初始化class里面的name
            output_lang = Lang(lang1)
        else:
            input_lang = Lang(lang1)
            output_lang = Lang(lang2)
        return input_lang, output_lang, pairs


def readTest(root, lang1='en', lang2='vi', reverse=False):     # 默认英语→其他语言，可以用reverse反转
    print('Reading test lines...')
    test_X_path = root + '/tst2012.en'
    test_Y_path = root + '/tst2012.vi'
    pairs = []
    with open(test_X_path, 'r', encoding='utf-8') as file_X:
        with open(test_Y_path, 'r', encoding='utf-8') as file_Y:
            lines_X = file_X.read().strip().replace('&amp; amp ; quot ;', '"')\
                .replace('&apos;', "'").replace('&quot;', '"')\
                .replace('&amp;', '&').replace('&#91;', '[')\
                .replace('&#93;', ']').replace('& amp ;', '&').split('\n')
            lines_Y = file_Y.read().strip().replace('&amp; amp ; quot ;', '"')\
                .replace('&apos;', "'").replace('&quot;', '"')\
                .replace('&amp;', '&').replace('&#91;', '[')\
                .replace('&#93;', ']').replace('& amp ;', '&').split('\n')
            for (line, line_) in zip(lines_X, lines_Y):
                if line != '':
                    pairs.append([normalizeString(line), normalizeString_(line_)])
        # print('Length of pairs:' + str(len(pairs)))
        if reverse:
            pairs = [list(reversed(i)) for i in pairs]
        return pairs


def filterPair(pairs):
    pair_filter = []
    for p in pairs:
        print(p)
        if len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH:
            pair_filter.append(p)
        # if True:        # 不过滤最大长度
        #     pair_filter.append([x[:MAX_LENGTH]for x in p])
    return pair_filter


def tensorFromSentence(lang, sentence):
    indexes = []
    for word in sentence.split(' '):
        try:
            indexes.append(lang.word2index[word])
        except:
            indexes.append(UNK_token)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)


def tensorFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):  # input_lang.n_words, hidden_size=256
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)  # embedding_dim=hidden_size, hidden_size, num_layers=1, bidirectional=False

    def forward(self, input, hidden):  # input.size()= torch.Size([1])
        embedded = self.embedding(input).view(1, 1, -1)  # 1 * hidden_size → 1 * 1 * hidden_size:torch.Size([1, 1, 256])
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)  # num_layers * num_directions, batch, hidden_size


# Attention 解码器
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.2, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)  # 1 * 1 * 256
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1  # (1 * 256与1 * 256) → Linear 1 * 10
        )
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(
            0))  # 1 * 1 * 10, 1 * max_length * encoder.hidden_size → 1 * 1 * 256
        output = torch.cat((embedded[0], attn_applied[0]), 1)  # 1 * 512
        output = self.attn_combine(output).unsqueeze(0)  # 1 * 1* 256
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size()[0]
    target_length = target_tensor.size()[0]

    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

    loss = 0

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = torch.tensor([SOS_token], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:     # Learn from reference
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])        # 1 * output_size
            decoder_input = target_tensor[di]       # Teacher forcing

    else:       # Learn from model
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.detach()  # detach from history as input
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


def trainIters(n_iters, encoder, decoder):
    print('training...')
    start = time.time()
    # n_iters = 130000 * 10
    print_every = 1000
    plot_every = 100
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=0.01)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=0.01)
    print(n_iters)
    training_pairs = [tensorFromPair(random.choice(pairs)) for _ in range(n_iters)]       # n_iters
    criterion = nn.NLLLoss()

    plot_losses = []

    print(1)

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]
        loss = train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss
        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('{} 当前iter={} 完成进度{:.3f}% Loss:{:.4f}'.format(timeSince(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_loss_total = 0
            plot_losses.append(plot_loss_avg)

    torch.save(encoder.state_dict(), '50/model_encoder_1.pth')
    torch.save(decoder.state_dict(), '50/model_decoder_1.pth')

    plt.figure()
    plt.plot(plot_losses)
    plt.savefig('50/loss_1.png')
    # plt.show()


def evaluate(encoder, decoder, sentence):
    max_length = MAX_LENGTH
    encoder.load_state_dict(torch.load('50/model_encoder_1.pth'))
    decoder.load_state_dict(torch.load('50/model_decoder_1.pth'))
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]

        encoder_hidden = encoder.initHidden()
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0][0]

        decoder_input = torch.tensor([[SOS_token]], device=device)
        decoder_hidden = encoder_hidden
        decoder_words = []

        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoder_words.append('<EOS>')
                break
            else:
                decoder_words.append(output_lang.index2word[topi.item()])
            decoder_input = topi.detach()

        return decoder_words, decoder_attentions[:di+1]


def test(encoder, decoder, test_pairs, path1, path2):
    max_length = MAX_LENGTH
    encoder.load_state_dict(torch.load('50/model_encoder_1.pth'))
    decoder.load_state_dict(torch.load('50/model_decoder_1.pth'))
    test_loss = []
    with torch.no_grad():
        test_criterion = nn.CrossEntropyLoss()
        with open(path1, 'w', encoding='utf-8') as source_f:
            with open(path2, 'w', encoding='utf-8') as result_f:
                _ = 0
                for pair in test_pairs:
                    tmp_loss = 0
                    source_f.write(pair[1] + '\n')
                    input_tensor, target_tensor = tensorFromPair(pair)
                    input_length = input_tensor.size()[0]
                    target_length = target_tensor.size()[0]

                    encoder_hidden = encoder.initHidden()
                    encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

                    for ei in range(input_length):
                        encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
                        encoder_outputs[ei] += encoder_output[0][0]

                    decoder_input = torch.tensor([[SOS_token]], device=device)
                    decoder_hidden = encoder_hidden
                    decoder_words = []

                    decoder_attentions = torch.zeros(max_length, max_length)

                    for di in range(max_length):
                        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden,
                                                                                    encoder_outputs)
                        if di < target_length:
                            tmp_loss += test_criterion(decoder_output, target_tensor[di])
                        decoder_attentions[di] = decoder_attention.data
                        topv, topi = decoder_output.data.topk(1)
                        if topi.item() == EOS_token:
                            # decoder_words.append('<EOS>')
                            break
                        else:
                            decoder_words.append(output_lang.index2word[topi.item()])
                        decoder_input = topi.detach()
                    test_loss.append(tmp_loss/min(di, target_length))
                    result_f.write(' '.join(decoder_words) + '\n')
                    if (_ + 1) % 10 == 0:
                        print('\r' + str(_ + 1), end='')
                    _ += 1
                print()
    print('Loss:' + str(sum(test_loss)/len(test_loss)) + ' ' + 'PPL:' + str(math.exp(sum(test_loss)/len(test_loss))))


if __name__ == '__main__':
    root = 'iwslt15_en_vt'
    input_lang, output_lang, pairs = readLangs(root, reverse=True)
    print('Read %d sentence pairs' % len(pairs))
    # print(pairs[-1])
    pairs = filterPair(pairs)
    print('Trimmed to %d sentence pairs' % len(pairs))
    n_iters = len(pairs) * 10
    # print('n_iters:' + str(n_iters))

    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    hidden_size = 256
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.2).to(device)

    teacher_forcing_ratio = 0.5

    trainIters(n_iters, encoder1, attn_decoder1)

    # sentence = 'Câu chuyện này chưa kết thúc .'
    # decoder_words, decoder_attentions = evaluate(encoder1, attn_decoder1, sentence)
    # print(' '.join(decoder_words))
    # sentence = 'Ông rút lui vào yên lặng .'
    # decoder_words, decoder_attentions = evaluate(encoder1, attn_decoder1, sentence)
    # print(' '.join(decoder_words))
    # sentence = 'Ông qua đời , bị lịch sử quật ngã .'
    # decoder_words, decoder_attentions = evaluate(encoder1, attn_decoder1, sentence)
    # print(' '.join(decoder_words))
    # sentence = 'Ông là ông của tôi .'
    # decoder_words, decoder_attentions = evaluate(encoder1, attn_decoder1, sentence)
    # print(' '.join(decoder_words))
    # sentence = 'Tôi chưa bao giờ gặp ông ngoài đời .'
    # decoder_words, decoder_attentions = evaluate(encoder1, attn_decoder1, sentence)
    # print(' '.join(decoder_words))
    #
    # test_pairs = readTest(root, reverse=True)
    # print('Read %d test sentence pairs' % len(test_pairs))
    # source = "ref_tst2012_I.txt"
    # result = "result_tst2012_I.txt"
    # test(encoder1, attn_decoder1, test_pairs, source, result)

    # print(file_bleu([source], result, verbose=True))
    # print(get_moses_multi_bleu(source, source, lowercase=False))

    # ref_sentence = ['this is not a finished story .',
    #                 'it is a jigsaw puzzle still being put together .']
    # result_sentence = ['this is the not . .',
    #                    'it s a game a screen , a game .']

    # print(list_bleu([ref_sentence], result_sentence))
    # print(get_moses_multi_bleu(result_sentence, ref_sentence, lowercase=True))
