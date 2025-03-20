import torch
from humanize import intword

from model import BigramLanguageModel
from config import GPTSmallConfig, GPT10MConfig

torch.manual_seed(1337)
config = GPTSmallConfig()

"""
Auxiliary functions
"""
with open("train.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
print("Vocab: " + "".join(chars).strip() + f" (size: {len(chars)})")

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # noqa: E731
decode = lambda l: "".join([itos[i] for i in l])  # noqa: E731, E741

data = torch.tensor(encode(text), dtype=torch.long)
n = int(config.training_split * len(data))
train_data = data[:n]
val_data = data[n:]

# Set up device for training
device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def get_batch(split: str) -> tuple[torch.Tensor, torch.Tensor]:
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - config.block_size, (config.batch_size,))
    x = torch.stack([data[i : i + config.block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + config.block_size + 1] for i in ix]).to(device)
    return x, y


xb, yb = get_batch("train")

if __name__ == "__main__":
    m = BigramLanguageModel(config)
    print(f"Model params: {intword(m.get_param_count())}")
    # print(f"Model params: {m.get_param_count() / 1e6:.2f}M")

    # Move model to device
    m = m.to(device)

    print("Before training:")
    print(
        decode(
            m.generate(idx=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[
                0
            ].tolist()
        )
    )

    # Move data to device
    train_data = train_data.to(device)
    val_data = val_data.to(device)

    logits, loss = m(xb, yb)
    optimiser = torch.optim.AdamW(m.parameters(), lr=config.learning_rate)
    for epoch in range(10000):
        # Get training batch and compute loss
        xb, yb = get_batch("train")
        logits, train_loss = m(xb, yb)
        optimiser.zero_grad(set_to_none=True)
        train_loss.backward()
        optimiser.step()

        # Get validation batch and compute loss
        if epoch % 100 == 0:
            xb, yb = get_batch("val")
            with torch.no_grad():
                logits, val_loss = m(xb, yb)
            print(
                f"Epoch {epoch}/{10000} | Train Loss: {train_loss.item():.4f} | Val Loss: {val_loss.item():.4f}"
            )

    print("After training")
    print(
        decode(
            m.generate(idx=torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[
                0
            ].tolist()
        )
    )
