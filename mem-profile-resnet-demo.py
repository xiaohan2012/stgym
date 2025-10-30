# (c) Meta Platforms, Inc. and affiliates.
import logging
import socket
from datetime import datetime

import torch
import torch.nn as nn

logging.basicConfig(
    format="%(levelname)s:%(asctime)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(level=logging.INFO)

TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"

# Keep a max of 100,000 alloc/free events in the recorded history
# leading up to the snapshot.
MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT: int = 100000


def start_record_memory_history() -> None:
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable. Not recording memory history")
        return

    logger.info("Starting snapshot record_memory_history")
    torch.cuda.memory._record_memory_history(
        max_entries=MAX_NUM_OF_MEM_EVENTS_PER_SNAPSHOT
    )


def stop_record_memory_history() -> None:
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable. Not recording memory history")
        return

    logger.info("Stopping snapshot record_memory_history")
    torch.cuda.memory._record_memory_history(enabled=None)


def export_memory_snapshot() -> None:
    if not torch.cuda.is_available():
        logger.info("CUDA unavailable. Not exporting memory snapshot")
        return

    # Prefix for file names.
    host_name = socket.gethostname()
    timestamp = datetime.now().strftime(TIME_FORMAT_STR)
    file_prefix = f"{host_name}_{timestamp}"

    try:
        logger.info(f"Saving snapshot to local file: {file_prefix}.pickle")
        torch.cuda.memory._dump_snapshot(f"{file_prefix}.pickle")
    except Exception as e:
        logger.error(f"Failed to capture memory snapshot {e}")
        return


class SimpleTransformer(nn.Module):
    """A simple transformer model for sequence classification."""

    def __init__(
        self,
        vocab_size=10000,
        d_model=512,
        nhead=8,
        num_layers=6,
        num_classes=10,
        max_seq_length=128,
    ):
        super().__init__()

        # Embedding layers
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, max_seq_length, d_model))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x shape: (batch_size, sequence_length)
        seq_length = x.size(1)

        # Embed and add positional encoding
        x = self.embedding(x)
        x = x + self.positional_encoding[:, :seq_length, :]

        # Pass through transformer
        x = self.transformer(x)

        # Global average pooling
        x = x.mean(dim=1)

        # Classify
        x = self.classifier(x)

        return x


# Simple Transformer example to demonstrate how to capture memory visuals.
def run_transformer(num_iters=5, device="cuda:0"):
    # Model parameters
    vocab_size = 10000
    batch_size = 8
    seq_length = 128
    num_classes = 10

    # Initialize model
    model = SimpleTransformer(
        vocab_size=vocab_size,
        d_model=512,
        nhead=8,
        num_layers=6,
        num_classes=num_classes,
        max_seq_length=seq_length,
    ).to(device=device)

    # Create sample inputs (random token IDs)
    inputs = torch.randint(0, vocab_size, (batch_size, seq_length), device=device)
    labels = torch.randint(0, num_classes, (batch_size,), device=device)

    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    # Start recording memory snapshot history
    start_record_memory_history()

    for iteration in range(num_iters):
        # Forward pass
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        # Backward pass
        loss.backward()

        # Optimizer step
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        logger.info(f"Iteration {iteration + 1}/{num_iters}, Loss: {loss.item():.4f}")

    # Create the memory snapshot file
    export_memory_snapshot()

    # Stop recording memory snapshot history
    stop_record_memory_history()


if __name__ == "__main__":
    # Run the transformer model
    run_transformer()
