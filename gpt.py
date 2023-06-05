import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention_Head(nn.Module):
    """
    A single self-attention head, using query, key and value
    """

    def __init__(self, embedding_n=32, head_size=32):
        super().__init__()
        self.query = nn.Linear(embedding_n, head_size, bias=False)
        self.key = nn.Linear(embedding_n, head_size, bias=False)
        self.value = nn.Linear(embedding_n, head_size, bias=False)

        self.head_size = head_size

    def forward(self, x):
        B, T = x.shape[0], x.shape[1]
        device = x.device

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        weight = k @ q.transpose(2, 1)

        mask = torch.ones(T, T).to(device)
        tril = torch.tril(mask)
        weight = weight.masked_fill(tril == 0,
                                    torch.tensor(float('-inf')).to(device))
        weight = torch.softmax(weight * (self.head_size**(-0.5)), dim=2)
        # finally, the logits
        attention = weight @ v

        return attention


class FeedForward(nn.Module):
    """
    A simple MLP-style module with the structure: Linear, ReLU, Linear
    """

    def __init__(self, embedding_n=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(embedding_n, 4 * embedding_n),
                                 nn.ReLU(),
                                 nn.Linear(4 * embedding_n, embedding_n))

    def forward(self, x):
        return self.net(x)


class MultiHead(nn.Module):
    """
    Combines multiple 'SelfAttention_Head' instances in a single class. Output of each selfAttention_head
    is concatenated, and then fed to a linear projection layer
    """

    def __init__(self, num_heads, embedding_n=32, head_size=32):
        super().__init__()
        self.heads = nn.ModuleList([
            SelfAttention_Head(embedding_n, head_size=head_size // num_heads)
            for _ in range(num_heads)
        ])
        self.projection = nn.Linear(head_size, head_size)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.projection(x)


class Block(nn.Module):
    """
    A block, as suggested in the "Attention is all you need" paper. Each block has:
    A MultiHead (attention) and a FeedForward module, both with skip connections
    """

    def __init__(self, embedding_n=32, num_heads=4, head_size=32):
        super().__init__()
        self.attention = MultiHead(num_heads=4,
                                   embedding_n=embedding_n,
                                   head_size=head_size)
        self.feed_forward = FeedForward(embedding_n=embedding_n)
        self.layer_norm1 = nn.LayerNorm(embedding_n)
        self.layer_norm2 = nn.LayerNorm(embedding_n)

    def forward(self, x):
        x = x + self.attention(self.layer_norm1(x))
        x = x + self.feed_forward(self.layer_norm2(x))
        return x


class GPT_Nano(nn.Module):
    """
    This is the GPT-style language model we are building. We are using token and positional embeddings,
    3 blocks (each containing multihead attention and feedforward network), and a linear output layer
    """

    def __init__(self,
                 vocab_size,
                 block_size=8,
                 embedding_n=32,
                 attention_head_size=32,
                 num_attention_heads=4,
                 num_layers=3):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_n)
        self.pos_embedding = nn.Embedding(block_size, embedding_n)
        self.block_size = block_size

        self.blocks = nn.Sequential(*[Block(embedding_n=embedding_n, num_heads=num_attention_heads, head_size=attention_head_size) for _ in range(num_layers)])

        self.layer_norm = nn.LayerNorm(embedding_n)
        self.linear_head = nn.Linear(embedding_n, vocab_size)

        # better initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, source, target=None):
        B, T = source.shape[0], source.shape[1]
        seq_embedding = self.token_embedding(source)
        pos_embedding = self.pos_embedding(
            torch.arange(source.shape[1], device=source.device))
        x = seq_embedding + pos_embedding

        x = self.blocks(x)

        x = self.layer_norm(x)
        logits = self.linear_head(x)

        B, T, C = logits.shape

        if target is not None:
            logits = logits.view(B * T, C)
            target = target.view(B * T)
            loss = F.cross_entropy(logits, target)
            logits = logits.view(B, T, C)

        else:
            loss = None

        return logits, loss

    def generate(self, source, max_len=10):
        for _ in range(max_len):
            # only use previous tokens up to the max of block_size
            s = source[:, -self.block_size:]

            logits, _ = self(s)
            # only use the last time-step prediction
            logits = logits[:, -1, :]

            probs = torch.softmax(logits, dim=1)

            # instead of picking the max probablitiy index (through argmax), we sample from the distribution
            next_char_prediction = torch.multinomial(probs, num_samples=1)

            source = torch.cat([source, next_char_prediction], dim=1)

        return source
