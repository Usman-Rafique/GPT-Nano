import torch
import torch.nn as nn
import torch.nn.functional as F
from gpt import GPT_Nano
import tqdm

# settings and hyperparameters
input_fname = 'input_shakespeare.txt'  # text file with training data

# We have a small model that can be quickly trained, and a scaled up model that gets good results.
# I am keeping both sets of hyperparameters here
model_type = 'small'  # 'small' or 'big'

if model_type == 'small':
    # a smaller network getting train loss 2.17 and val loss 1.92
    block_size = 8  # maximum context size
    embedding_n = 32  # embedding dimension for the model
    num_attention_heads = 4  # number of heads in multihead self-attetion
    num_layers = 3  # number of Blocks
    lr = 3e-4
    batch_size = 64
    dropout = 0.0
elif model_type == 'big':
    # deeper network, currently getting train loss 1.2, val loss 0.54
    # ToDo: add expected train and val loss
    block_size = 256  # maximum context size
    embedding_n = 384  # embedding dimension for the model
    num_attention_heads = 6  # number of heads in multihead self-attetion
    num_layers = 6  # number of Blocks
    lr = 3e-4
    batch_size = 64
    dropout = 0.2

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
num_train_iterations = 5000

torch.manual_seed(1337)

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
    source = torch.stack(
        [torch.tensor(encoded_data[i:i + block_size]) for i in indices],
        dim=0).to(device)
    target = torch.stack(
        [torch.tensor(encoded_data[i + 1:i + block_size + 1]) for i in indices],
        dim=0).to(device)

    return source, target


# model
model = GPT_Nano(vocab_size=vocab_size,
                 block_size=block_size,
                 embedding_n=embedding_n,
                 num_attention_heads=num_attention_heads,
                 num_layers=num_layers,
                 dropout=dropout)
model.to(device)
print('Number of paramaters (M):',
      sum(p.numel() for p in model.parameters()) / 1e6)
init_vals = torch.tensor(encode('A')).unsqueeze(0).to(device)
generated = model.generate(init_vals, 100)[0]
print('random generation before any training:')
print(decode(generated.tolist()))


# val loss calulation
@torch.no_grad()
def calculate_val_loss(num_iterations=1000):
    model.eval()
    val_loss = 0
    for _ in range(num_iterations):
        source, target = get_batch('val', device=device)
        logits, loss = model(source, target)
        val_loss += loss.item()

    val_loss /= num_iterations
    model.train()
    return val_loss


model = torch.compile(model)

print(f'starting training on {device=}')

# Training
optim = torch.optim.AdamW(model.parameters(), lr=lr)
train_loss = 0

min_val_loss = float('inf')

pbar = tqdm.trange(num_train_iterations)
for i in pbar:
    optim.zero_grad(set_to_none=True)
    source, target = get_batch('train', device=device, batch_size=batch_size)

    logits, loss = model(source, target)

    loss.backward()
    optim.step()

    train_loss += loss.item()

    # print val loss 10 times during training
    if i % (num_train_iterations // 100) == 0 or i == num_train_iterations - 1:
        val_loss = calculate_val_loss()
        pbar.set_description(f'train loss:{(train_loss)/(i+1):.4f}, {val_loss=:.4f}')
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            fname = 'model_weights_' + model_type + '_' + input_fname + '_best.pth'
            torch.save(model.state_dict(), fname)

# save checkpoint, just in case
fname = 'model_weights_' + model_type + '_' + input_fname + '.pth'
torch.save(model.state_dict(), fname)
print('finished training')

# print model generation results
print('Generation after training:')

for _ in range(5):
    start_char = torch.randint(0, vocab_size, (1,))
    init_vals = start_char.unsqueeze(0).to(device)
    generated = model.generate(init_vals, 200)[0]
    print('===============================\n', decode(generated.tolist()))

# save results to a text file
start_char = torch.zeros((1,)).long()
init_vals = start_char.unsqueeze(0).to(device)
fname_output = 'generated_text_' + model_type + '_' + input_fname + '.pth'
with open(fname_output, 'w') as f:
    generated = model.generate(init_vals, 5000)[0]
    f.write(decode(generated.tolist()))
