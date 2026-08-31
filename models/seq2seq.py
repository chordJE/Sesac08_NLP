#[seq2seq의 장점과 핵심 아이디어]
#1. 기존 모델 한계
#1-1. 기존 모델은 출력이 1:1 고정이 되어 있었음(input->output)
#1-2. 어순과 구조가 맞지 않는 언어의 번역 문제 / QA  => input을 읽고 바로 output을 낼 수 있도록 하는 구조 필요
#2. 아이디어 -> "인코더 + 디코더" (입력/출력 길이를 분리)

#[모델 훈련 연습]
#1. 데이터 준비
#2. 데이터셋 클래스 만들기
#3. seq2seq 모델 만들기
#4. 훈련-검증

import re, unicodedata

#텍스트 정제
def normalize(text, lang='en'):
    if lang=='fr':
        #프랑스어 악센트 기호 처리
        text = unicodedata.normalize('NFD', text)
        #아스키 코드 범위 넘으면 'ignore' 한 후, 다시 ascii 변환
        text = text.encode('ascii', 'ignore').decode('ascii')

    text = text.lower().strip()
    text = re.sub(r'([.!?])', r' \1', text) #., !, ? 앞을 한 칸 띄우세요
    text = re.sub(r'[^a-z.!? ]+', ' ', text) #a-z.!?공백 이 아닌 것을 ' '공백으로 변환
    return text.split()


import os, shutil, urllib.request , zipfile
from collections import Counter
import torch
import random

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

#인터넷에서 데이터를 다운로드
def load_data(max_pairs=20000):
    data_file = './data/data/eng-fra.txt'
    if not os.path.exists(data_file):
        #다운로드가 안됐다면? -> 다운로드 하세요!
        os.makedirs('data', exist_ok=True)

        urllib.request.urlretrieve(
            'https://download.pytorch.org/tutorial/data.zip', 'data/data.zip'
        )
        with zipfile.ZipFile('data/data.zip') as z :
            z.extractall('data')

    pairs = []
    with open('./data/data/eng-fra.txt', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            print(parts)
            if len(parts) < 2 :
                continue

            en = normalize(parts[0], 'en')
            fr = normalize(parts[1], 'fr')

            pairs.append((en, fr))

    import random
    random.shuffle(pairs)
    pairs = pairs[:max_pairs]
    return pairs


#Vocab 만들기
def build_vocab(pairs):
    en_cnt, fr_cnt = Counter(), Counter()
    for en, fr in pairs:
        en_cnt.update(en)
        fr_cnt.update(fr)
    #영어 Vocab
    #프랑스어 Vocab
    #MAX_LEN = 30
    en_vocab, fr_vocab = Vocab(30), Vocab(30)

    #w-> 단어, f->빈도
    for w, f in en_cnt.items():
        if f >= 2:
            en_vocab.add(w)

    for w, f in fr_cnt.items():
        if f >= 2:
            fr_vocab.add(w)

    print(f'영어 어휘 : {len(en_vocab)}, 프랑스어 어휘 {len(fr_vocab)}')
    return en_vocab, fr_vocab

class Vocab:
    def __init__(self, max_len):
        #w(word) i(index) w2i -> 단어를 숫자로 / i2w -> 숫자를 단어로
        self.w2i = {'<PAD>':0, '<SOS>':1, '<EOS>':2, '<UNK>':3}
        self.i2w = {v:k for k, v in self.w2i.items()}
        self.MAX_LEN = max_len
    #vocab을 추가함
    def add(self, word):
        if word not in self.w2i:
            i = len(self.w2i)
            self.w2i[word] = len(self.w2i)
            self.i2w[i] = word

    #토큰 인코드(문자->숫자)
    def encode(self, tokens):
        # <SOS> [그, 는, 말하다, 피곤하다고] <EOS>
        SOS, EOS, UNK = 1, 2, 3
        ids = [SOS] + [self.w2i.get(t, UNK) for t in tokens] + [EOS]

        #MAX_LEN(최장단어의 길이) 보다 작은 경우 아래처럼 0번 더해줌(패딩)
        #MAX_LEN = 50이라고 가정, 
        # 50 - len(ids) +2(sos, eos)
        ids += [0] * (self.MAX_LEN -len(ids) +2)
        return ids[:self.MAX_LEN +2]

    #숫자->문자
    def decode(self, ids):
        out = [] #id가 변환되어 쌓일 문자열 리스트

        for i in ids:
            w = self.i2w.get(i, '<UNK>')
            # <sos> 나는 이렇게 말했다 <pad> <pad> <eos>
            if w in ('<PAD>', '<SOS>', '<EOS>'):
                continue #append하지 말고 넘어가라.
            out.append(w)
        return out

    def __len__(self): 
        return len(self.w2i)

#LSTM
class Encoder(nn.Module):
    def __init__(self, vocab_size, 
                 embed_dim=128, 
                 hidden_size=256,
                 num_layers = 5 ,
                 dropout=0.3):
        super().__init__()
        #vocab_size, embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim) 
        self.lstm = nn.LSTM(
            embed_dim, 
            hidden_size,
            num_layers = num_layers,
            batch_first = True,
            dropout = dropout
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embed = self.embedding(x)
        dropout = self.dropout(embed)
        _, (hidden, cell) = self.lstm(dropout)
        return hidden, cell

        # lstm층을 통과한 결과물은? 
        # out, _ = self.lstm(embedding)
        # last = out[:, -1, :]
        # return self.fc(self.dropout(last))

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size, 
                 num_layers, dropout, PAD_IDX):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, 
                                      padding_idx=PAD_IDX)
        self.ltsm = nn.LSTM(embed_dim, hidden_size,
                            num_layers,
                            batch_first=True,
                            dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        #디코더 -> 결과 출력
        self.fc = nn.Linear(hidden_size, vocab_size)

    #token  ->(번역할 X)
    #hidden ->(from encoder)
    #cell   ->(from encoder)
    def forward(self, token, hidden, cell):
        #Token => (B) -> 언스퀴즈 (B, 1) -> 임베딩 거친 후 (B, 1, E)
        embed = self.dropout(self.embedding(token.unsqueeze(1)))
        #(hidden, cell) 은 인코더의 출력물, 전체 데이터에 대한 정보 기억
        #(hidden, cell)은 과거의 기억(hidden state) -> 앞으로 나올 단어 산출
        out, (hidden, cell) = self.ltsm(embed, (hidden, cell))
        word = self.fc(out.squeeze(1))
        return word, hidden, cell

class seq2seq(nn.Module):
    def __init__(self, encoder, decoder, trg_vocab_size, device='cuda'):
        super().__init__()
        self.encoder = encoder 
        self.decoder = decoder
        self.trg_vocab_size = trg_vocab_size
        self.device = device

    #seq2seq의 훈련
    def forward(self, src, trg, ratio):
        B, trg_len = trg.size()

        outputs = torch.zeros(B, trg_len, self.trg_vocab_size, device=self.device)

        hidden, cell = self.encoder(src)
        token = trg[:, 0]

        for t in range(trg_len):
            logit, hidden, cell = self.decoder(token, hidden, cell)
            outputs[:, t] = logit
            token = trg[:, t] if random.random() < ratio else logit.argmax(1)

        return outputs



    #데코레이터! @ with~ 이번에 새로 정의하는 함수가 다른 함수의 기능을 이어받았으면 좋겠다!
    #with torch.no_grad():
    @torch.no_grad()
    def translate(self, src, max_len):
        self.eval()
        hidden, cell = self.encoder(src)

        #0(SOS)으로 시작하는 빈 텐서 만들어놓기
        token = torch.tensor([SOS_IDX], device=self.device)
        result = []

        for _ in range(max_len):
            logit, hidden, cell = self.decoder(token, hidden, cell)
            temp = logit.argmax(1) #temp->전체 vocab중 가장높은 확률을 가진 vocab 1개
            if temp.item() == EOS_IDX :
                break
            result.append(temp.item())

        return result

PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX = 0, 1, 2, 3


# 영어-프랑스어 쌍 (영어, 프랑스어)
class TranslationDataset(Dataset):
    def __init__(self, pairs, src_vocab, trg_vocab):
        self.data = [
            (torch.tensor(src_vocab.encode(en), dtype=torch.long), 
             torch.tensor(trg_vocab.encode(fr), dtype=torch.long)) 
                    for en, fr in pairs
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def run_epoch(model, loader, optimizer, criterion, device, train=True, tf=0.5):
    model.train() if train else model.eval()
    total_loss = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for src, trg in loader:
            src, trg = src.to(device), trg.to(device)
            out = model(src, trg, ratio=tf if train else 0.0)
            loss = criterion(out[:, 1:].reshape(-1, out.size(-1)),
                             trg[:, 1:].reshape(-1))
            if train:
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item()

    return total_loss / len(loader)

#arg_parse() 
import time
def train_model(model, train_loader, valid_loader, lr, epochs, device):
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)
    history   = {'train': [], 'valid': []}
    best_loss = float('inf')

    for epoch in range(epochs):
        tf  = max(0.3, 0.9 - epoch * 0.06)   # 점진적으로 teacher forcing 비율 감소
        t0  = time.time()
        tl  = run_epoch(model, train_loader, optimizer, criterion, device, train=True,  tf=tf)
        vl  = run_epoch(model, valid_loader, optimizer, criterion, device, train=False)
        scheduler.step(vl)
        history['train'].append(tl); history['valid'].append(vl)

        if vl < best_loss:
            best_loss = vl
            torch.save(model.state_dict(), 'seq2seq_basic_best.pt')

        print(f'Epoch {epoch+1:2d}/{epochs} | '
              f'train {tl:.4f} | valid {vl:.4f} | tf {tf:.2f} | {time.time()-t0:.1f}s')

    return history



if __name__ == '__main__':
    pairs = load_data()
    print(f'페어 길이 : {len(pairs)}, 페어[0] {pairs[0]}')

    en_vocab, fr_vocab = build_vocab(pairs)

    #클래스 정의 -> vocab, encoder, decoder, seq2seq 
    #훈련->추론 => 데이터준비, 데이터로더, 인코더/디코더/모델(생성), train, 추론(pred)

    #1.커스텀데이터셋으로 en_vocab, fr_vocab의 데이터셋 제작
    dataset = TranslationDataset(pairs, en_vocab, fr_vocab)

    n = len(dataset)
    n_train = int(n * 0.8)
    n_valid = int(n * 0.1)
    n_test = int(n*0.1)

    train_set, valid_set, test_set = random_split(dataset, 
                                                  [n_train, n_valid, n_test])

    #2. 데이터 로더
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=16)
    test_loader = DataLoader(test_set, batch_size=16)

    #3. 훈련에 필요한 인코더/디코더/모델 객체 생성
    #주의할 점!! hidden을 공유하므로 HIDDEN_SIZE와 NUM_LAYERS는 같은 숫자여야 함.
    EMBED_DIM = 256
    HIDDEN_SIZE = 256
    NUM_LAYERS = 5
    DROPOUT = 0.2

    #영->프 / 프->영 en_vocab, fr_vocab
    encoder = Encoder(len(en_vocab))
    decoder = Decoder(len(fr_vocab), 
                    embed_dim = EMBED_DIM, 
                    hidden_size = HIDDEN_SIZE, 
                    num_layers = NUM_LAYERS, 
                    dropout = DROPOUT, 
                    PAD_IDX= PAD_IDX)

    #디바이스셋업먼저
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = seq2seq(encoder, decoder, len(fr_vocab)).to(device)

    #4.train 시작
    history = train_model(model, 
                          train_loader, 
                          valid_loader,
                          0.01,
                          100,
                          device=device)

    #5.실제 번역(추론)
    #model.translate()