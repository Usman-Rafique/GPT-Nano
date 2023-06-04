import torch
import torch.nn as nn
import torch.nn.functional as F
from models import Bigram_Model, Language_Model
# settings and hyperparameters
input_fname = 'input.txt'  # text file with training data
block_size = 8  # maximum context size
embedding_n = 32  # embedding dimension for the model
attention_head_size = 32  # head size for self-attention
num_attention_heads = 4
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
num_train_iterations = 10000
lr = 1e-3
batch_size = 256

# read data
with open(input_fname, 'r') as f:
    data = f.read()

vocabulary = sorted(list(set(data)))
print(f'{vocabulary=}')
vocab_size = len(vocabulary)
print(f'{vocab_size=}')

# projecting characters to integer indices
stoi = {c: i for i, c in enumerate(vocabulary)}

# indices to characters
itos = {i: c for i, c in enumerate(vocabulary)}


def encode(string):
    return [stoi[c] for c in string]


def decode(indices):
    return ''.join(itos[i] for i in indices)


# make train/val splits
encoded_data = encode(data)
train_data = data[:int(0.9 * len(encoded_data))]
val_data = data[int(0.9 * len(encoded_data)):]

print('train size:', len(train_data))
print('val size:', len(val_data))


# dataloader
def get_batch(mode='train', batch_size=4, device='cpu'):
    data = train_data if mode == 'train' else val_data
    indices = torch.randint(len(data) - block_size, (batch_size,))
    source = torch.stack([torch.tensor(encoded_data[i:i + 8]) for i in indices],
                         dim=0).to(device)
    target = torch.stack(
        [torch.tensor(encoded_data[i + 1:i + 8 + 1]) for i in indices],
        dim=0).to(device)

    return source, target


# model
model = Language_Model(vocab_size=vocab_size,
                       block_size=block_size,
                       embedding_n=embedding_n,
                       attention_head_size=attention_head_size,
                       num_attention_heads=num_attention_heads)
model.to(device)

init_vals = torch.tensor(encode('A')).unsqueeze(0).to(device)
generated = model.generate(init_vals, 100)[0]
print('random generation before any training:')
print(decode(generated.tolist()))

# training

optim = torch.optim.AdamW(model.parameters(), lr=lr)


@torch.no_grad()
def calculate_val_loss(num_iterations=1000):
    val_loss = 0
    for _ in range(num_iterations):
        source, target = get_batch('val', device=device)
        logits, loss = model(source, target)
        val_loss += loss.item()

    val_loss /= num_iterations
    return val_loss


train_loss = 0

for i in range(num_train_iterations):
    optim.zero_grad()
    source, target = get_batch('train', device=device, batch_size=batch_size)

    logits, loss = model(source, target)

    loss.backward()
    optim.step()

    train_loss += loss.item()

    # print val loss 10 times during training
    if i % (num_train_iterations // 10) == 0:
        val_loss = calculate_val_loss()
        print(f'train loss, {(train_loss)/(i+1)}, {val_loss=}')

# save checkpoint, just in case
fname = 'model_weights.pth'
torch.save(model.state_dict(), fname)
print('finished training')

# print model generation results
print('Generation after training:')

for _ in range(5):
    start_char = torch.randint(0, vocab_size, (1,))
    init_vals = start_char.unsqueeze(0).to(device)
    generated = model.generate(init_vals, 200)[0]
    print('===============================\n', decode(generated.tolist()))
