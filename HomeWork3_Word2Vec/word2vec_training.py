
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from word2vec_dataset import Word2VecDataset
from word2vec_verification import Word2VecSGNS


def train_model(
    corpus,
    epochs=10,
    batch_size=16,
    embed_dim=50,
    lr=0.01,
    window_size=2,
    num_negatives=5,
):
    """Train a custom Word2Vec model using SGNS and manual gradients."""

    print("--- Initializing Word2Vec Training Pipeline ---")

    # 1. Initialize the custom dataset
    dataset = Word2VecDataset(
        raw_text_list=corpus,
        window_size=window_size,
        num_negatives=num_negatives,
        min_count=1,
        subsample_threshold=1e-3,
    )

    # Stop if no training pairs were generated
    if len(dataset) == 0:
        print("ERROR: The dataset contains no training pairs.")
        print("Check tokenization, vocabulary creation, and subsampling.")
        return None

    # 2. Create the DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    print(f"Vocabulary size: {dataset.vocab_size}")
    print(f"Training pairs: {len(dataset)}")
    print(f"Batches per epoch: {len(dataloader)}")

    # 3. Initialize the Word2Vec SGNS model
    model = Word2VecSGNS(
        vocab_size=dataset.vocab_size,
        embed_dim=embed_dim,
    )

    # 4. Initialize the optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("\n--- Starting Training ---")

    # 5. Train for the specified number of epochs
    for epoch in range(1, epochs + 1):
        model.train()

        running_loss = 0.0
        total_samples = 0

        for target_batch, context_batch, negative_batch in dataloader:

            # Clear gradients from the previous optimization step
            optimizer.zero_grad(set_to_none=True)

            # Forward pass: calculate the SGNS loss
            loss = model(
                target_batch,
                context_batch,
                negative_batch,
            )

            # Validate the loss
            if loss is None:
                raise RuntimeError(
                    f"Model returned None at epoch {epoch}."
                )

            if loss.ndim != 0:
                raise ValueError(
                    "Expected a scalar loss from the model."
                )

            if not torch.isfinite(loss).item():
                raise RuntimeError(
                    f"Non-finite loss at epoch {epoch}: {loss.item()}"
                )

            # 6. Calculate gradients using custom manual backpropagation
            grad_W_target, grad_W_context = model.manual_backward(
                target_batch,
                context_batch,
                negative_batch,
            )

            # 7. Assign manual gradients to the embedding parameters
            model.W_target.weight.grad = grad_W_target
            model.W_context.weight.grad = grad_W_context

            # 8. Update the embedding parameters using Adam
            optimizer.step()

            # 9. Accumulate the loss for reporting
            current_batch_size = target_batch.size(0)
            running_loss += loss.item() * current_batch_size
            total_samples += current_batch_size

        # Calculate the average loss for this epoch
        epoch_loss = running_loss / total_samples

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"-> Average SGNS Loss: {epoch_loss:.5f}"
        )

    print("\nTraining complete!")

    return model


if __name__ == "__main__":

    # Example natural language corpus about artificial intelligence
    wikipedia_ai_corpus = [
        "Artificial intelligence is intelligence demonstrated by machines which can learn "
        "from data solve complex problems and perform human tasks. The field of artificial "
        "intelligence was founded in 1956. This field experienced multiple waves of "
        "optimism followed by periods of loss known as ai winter.",

        "Today modern artificial intelligence relies heavily on deep learning and neural "
        "networks. Using deep learning systems computers can be trained to perform specific "
        "tasks by processing large amounts of data. Machine learning is a major part of "
        "artificial intelligence.",

        "Machine learning focuses on the development of algorithms that allow computers to "
        "learn from data and make predictions. Deep learning is a specialized type of machine "
        "learning inspired by neural networks. These deep neural networks can process data "
        "in complex ways enabling systems to perform human tasks.",

        "Artificial intelligence networks and systems can learn from data and perform human "
        "tasks. Today artificial intelligence is transforming industry by automating tasks. "
        "Most artificial intelligence examples rely on deep learning machine learning and "
        "data processing. Using these systems computers can process data and learn from the data.",

        "The development of deep learning methods has revived interest in the field of "
        "artificial intelligence. Modern artificial intelligence relies on computers large "
        "amounts of data and algorithms to learn from data solve complex problems and automate tasks.",

        "These computers process large amounts of data to allow systems to learn from data. "
        "Machine learning algorithms and neural networks allow artificial intelligence systems "
        "to perform tasks. Modern deep learning systems can process data solve complex "
        "problems and perform human tasks. The field of artificial intelligence relies on "
        "data processing to train these computers and algorithms.",
    ]

    # Run the training pipeline
    trained_model = train_model(
        corpus=wikipedia_ai_corpus,
        epochs=10,
        batch_size=16,
        embed_dim=25,
        lr=0.01,
        window_size=3,
        num_negatives=5,
    )

    if trained_model is not None:
        print("Model is ready for evaluation.")
