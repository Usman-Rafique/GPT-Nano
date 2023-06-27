import sys
import torch
from gpt import GPT_Nano
import tqdm
from omegaconf import OmegaConf


# wrap the code in a train function
def train(args):
    # read command line arguments using argparse
    cli_config = OmegaConf.from_cli(args)

    # find configuration file, default is configs/small_model.yaml
    if 'config_file' not in cli_config:
        config_file = 'configs/small_model.yaml'
    else:
        config_file = cli_config['config_file']

    # read config file
    config_from_file = OmegaConf.load(config_file)

    # merge both configurations, let the command line arguments override
    cfg = OmegaConf.merge(config_from_file, cli_config)
    print('Configuration:', cfg)

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    #num_train_iterations = 5000

    torch.manual_seed(1337)

    # read data
    with open(cfg.input_fname, 'r') as f:
        data = f.read()

    vocabulary = sorted(list(set(data)))
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
    def get_batch(mode='train',
                  batch_size=4,
                  block_size=cfg.block_size,
                  device='cpu'):
        data = train_data if mode == 'train' else val_data
        indices = torch.randint(len(data) - block_size, (batch_size,))
        source = torch.stack(
            [torch.tensor(encoded_data[i:i + block_size]) for i in indices],
            dim=0).to(device)
        target = torch.stack([
            torch.tensor(encoded_data[i + 1:i + block_size + 1])
            for i in indices
        ],
                             dim=0).to(device)

        return source, target

    # model
    model = GPT_Nano(vocab_size=vocab_size,
                     block_size=cfg.block_size,
                     embedding_n=cfg.embedding_n,
                     num_attention_heads=cfg.num_attention_heads,
                     num_layers=cfg.num_layers,
                     dropout=cfg.dropout)
    model.to(device)
    print('Number of paramaters (M):',
          sum(p.numel() for p in model.parameters()) / 1e6)

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

    if cfg.compile_model:
        model = torch.compile(model)

    print(f'starting training on {device=}')
    # Training
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    train_loss = 0
    min_val_loss = float('inf')

    pbar = tqdm.trange(cfg.num_train_iterations)
    for i in pbar:
        optim.zero_grad(set_to_none=True)
        source, target = get_batch('train',
                                   device=device,
                                   batch_size=cfg.batch_size)

        logits, loss = model(source, target)

        loss.backward()
        optim.step()

        train_loss += loss.item()

        # print val loss 10 times during training
        if i % (cfg.num_train_iterations //
                100) == 0 or i == cfg.num_train_iterations - 1:
            val_loss = calculate_val_loss()
            pbar.set_description(
                f'train loss:{(train_loss)/(i+1):.4f}, {val_loss=:.4f}')
            if val_loss < min_val_loss:
                min_val_loss = val_loss
                fname = 'model_weights_' + cfg.model_name + '_' + cfg.input_fname.split(
                    '/')[-1] + '_best.pth'
                torch.save(model.state_dict(), fname)

    # save checkpoint, just in case
    fname = 'model_weights_' + cfg.model_name + '_' + cfg.input_fname.split(
        '/')[-1] + '.pth'
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
    fname_output = 'generated_text_' + cfg.model_name + '_' + cfg.input_fname.split(
        '/')[-1] + '.pth'
    with open(fname_output, 'w') as f:
        generated = model.generate(init_vals, 5000)[0]
        f.write(decode(generated.tolist()))


# check if this is the file called from terminal, and read it's arguments
if __name__ == '__main__':
    # call the train function with the command line arguments
    train(sys.argv[1:])
