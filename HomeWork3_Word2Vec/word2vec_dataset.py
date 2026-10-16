import numpy as np
import torch
from torch.utils.data import Dataset
from collections import Counter

class Word2VecDataset(Dataset):
    def __init__(self, raw_text_list, window_size=5, num_negatives=5, min_count=5, subsample_threshold=1e-3):
        """
        Args:
            raw_text_list: List of strings (sentences/paragraphs) representing the corpus.
            window_size: Maximum sliding window radius.
            num_negatives: Number of noise words to sample per positive pair.
            min_count: Drop words appearing fewer times than this threshold.
            subsample_threshold: Parameter 't' in Mikolov's subsampling formula.
        """
        self.window_size = window_size
        self.num_negatives = num_negatives
        self.min_count = min_count
        self.subsample_threshold = subsample_threshold
        
        # 1. Tokenize and count tokens
        print("Tokenizing corpus...")
        self.tokenized_corpus = [sentence.lower().split() for sentence in raw_text_list]
        self.flat_tokens = [word for sentence in self.tokenized_corpus for word in sentence]
        self.total_words_raw = len(self.flat_tokens)
        
        # 2. Build Vocabulary maps
        self.word_counts = Counter(self.flat_tokens)
        self._build_vocab()
        
        # 3. Apply Subsampling to down-sample frequent stop words
        print("Applying subsampling to frequent tokens...")
        self.subsampled_corpus = self._apply_subsampling()
        
        # 4. Generate positive training pairs (target, context)
        print("Generating positive training contexts...")
        self.pairs = self._generate_pairs()
        
        # 5. Build Negative Sampling distribution vector
        print("Initializing negative sampling table...")
        self.neg_sampling_distribution = self._build_neg_sampling_distribution()

    def _build_vocab(self):
        """
        Filters out tokens based on min_count, then populates:
        - self.word2idx: dict maps word -> index
        - self.idx2word: list/dict maps index -> word
        - self.vocab_size: int total valid tokens
        """
        vocab_words = [
        word for word, count in self.word_counts.items()
        if count >= self.min_count
        ]

        self.word2idx = {
            word: idx for idx, word in enumerate(vocab_words)
        }

        self.idx2word = {
            idx: word for word, idx in self.word2idx.items()
        }   

        self.vocab_size = len(self.word2idx)

    def _apply_subsampling(self):
        """
        Implements Mikolov's subsampling logic.
        """

        subsampled_data = []

        for sentence in self.tokenized_corpus:

            filtered_sentence = []

            for word in sentence:

                # Skip words removed by min_count
                if word not in self.word2idx:
                    continue

                freq = self.word_counts[word] / self.total_words_raw

                keep_prob = (
                    np.sqrt(self.subsample_threshold / freq)
                    + (self.subsample_threshold / freq)
             )

                keep_prob = min(1.0, keep_prob)

                if np.random.rand() < keep_prob:
                    filtered_sentence.append(word)

            if len(filtered_sentence) > 1:
                subsampled_data.append(filtered_sentence)

        return subsampled_data


    def _generate_pairs(self):
        """
        Iterates over the subsampled corpus using a sliding window.
        """

        pairs = []

        for sentence in self.subsampled_corpus:

            indices = [
                self.word2idx[word]
                for word in sentence
            ]

            for center_pos in range(len(indices)):

                dynamic_window = np.random.randint(
                    1,
                    self.window_size + 1
                )

                start = max(0, center_pos - dynamic_window)
                end = min(
                    len(indices),
                    center_pos + dynamic_window + 1
                )

                target_idx = indices[center_pos]

                for context_pos in range(start, end):
                    if context_pos == center_pos:
                        continue

                    context_idx = indices[context_pos]

                    pairs.append(
                        (target_idx, context_idx)
                    )

        return pairs
    

    def _build_neg_sampling_distribution(self):
        probs = torch.zeros(self.vocab_size,   
                            dtype=torch.float32)

        for word, idx in self.word2idx.items():
            probs[idx] = self.word_counts[word] ** 0.75

        probs /= probs.sum()

        return probs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        """
        Fetches an item from the precomputed dataset pairs and appends negative samples dynamically.
        
        Returns:
            target_idx: torch.tensor (scalar long)
            context_idx: torch.tensor (scalar long)
            neg_indices: torch.tensor (shape: [num_negatives], dtype: long)
        """
        target_idx, context_idx = self.pairs[idx]
        negatives = []
        while len(negatives) < self.num_negatives:
            sample = torch.multinomial(
                self.neg_sampling_distribution,
                1
            ).item()
            if sample != context_idx:
                negatives.append(sample)

        neg_indices = torch.tensor(negatives, dtype=torch.long)
        
        return (
            torch.tensor(target_idx, dtype=torch.long),
            torch.tensor(context_idx, dtype=torch.long),
            neg_indices
        )

if __name__ == "__main__":
    dummy_corpus = [
        "The quick brown fox jumps over the lazy dog",
        "NLP students love vector representations and words like king and queen",
        "Deep learning frameworks compute gradients automatically but manual gradients build character"
    ]
    dataset = Word2VecDataset(dummy_corpus, window_size=2, num_negatives=3, min_count=1)
    print(f"\nSuccessfully built Dataset Structure.")
    print(f"Vocab Size: {dataset.vocab_size}")
    print(f"Total pairs: {len(dataset)}")
    if len(dataset) > 0:
        t, c, n = dataset[0]
        print(f"Sample Layout -> Target: {t.item()}, Context: {c.item()}, Negatives: {n.tolist()}")
