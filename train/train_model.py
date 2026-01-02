'''
    Main Archeticture
    
'''

import torch, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from datasets import Dataset as HuggingFaceDataset
from torch.utils.data import Dataset


class EncodingDecoding:
    def encode(self, text):
        print("encoding", text)
        return list(text.encode("utf-8"))

    def decode(self, indices):
        return bytes(indices).decode("utf-8", errors="ignore")

    @property
    def vocab_size(self):
        return 256

        
class CustomDataset(Dataset):
    def __init__(self, encoded_data, block_size=None):
        self.data = encoded_data
        self.block_size = block_size
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Truncate if longer than block_size
        if self.block_size is not None and len(item) > self.block_size:
            item = item[:self.block_size]
        
        return {
            'input_ids': item,
            'labels': item.clone()
        }

def collate_fn(batch):
    """Collate function to pad sequences to the same length."""
    input_ids = [item['input_ids'] for item in batch]
    labels = [item['labels'] for item in batch]
    
    # Find max length in batch
    max_len = max(len(seq) for seq in input_ids)
    
    # Pad sequences
    padded_input_ids = []
    padded_labels = []
    
    for inp, lab in zip(input_ids, labels):
        pad_length = max_len - len(inp)
        if pad_length > 0:
            # Pad with 0 (or you could use a special padding token)
            padded_inp = torch.cat([inp, torch.zeros(pad_length, dtype=inp.dtype)])
            padded_lab = torch.cat([lab, torch.zeros(pad_length, dtype=lab.dtype)])
        else:
            padded_inp = inp
            padded_lab = lab
        padded_input_ids.append(padded_inp)
        padded_labels.append(padded_lab)
    
    return {
        'input_ids': torch.stack(padded_input_ids),
        'labels': torch.stack(padded_labels)
    }

def compute_metrics(eval_pred):
    predictions, labels = eval_pred

    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    pred_tokens = np.argmax(predictions, axis=-1)

    mask = labels != -100

    correct = (pred_tokens == labels) * mask
    accuracy = correct.sum() / mask.sum()

    loss_fct = torch.nn.CrossEntropyLoss()
    predictions_tensor = torch.from_numpy(predictions[:, :-1, :])
    labels_tensor = torch.from_numpy(labels[:, 1:])

    shift_logits = predictions_tensor.reshape(-1, predictions_tensor.shape[-1])
    shift_labels = labels_tensor.reshape(-1)

    loss = loss_fct(shift_logits, shift_labels).item()
    perplexity = np.exp(loss)

    return {
        "accuracy": float(accuracy),
        "perplexity": float(perplexity),
        "loss": float(loss)
    }



def split_conversations(conversation, test_size=0.1, random_state=42):
    train_conv, test_conv = train_test_split(
        conversation,
        test_size=test_size,
        random_state=random_state
    )
    
    train_dataset = HuggingFaceDataset.from_dict({"text": train_conv})
    test_dataset = HuggingFaceDataset.from_dict({"text": test_conv})
    
    return train_dataset, test_dataset


def save_encoder(encoder, filepath="encoder_vocab.pt"):
    torch.save({
        'stoi': encoder.stoi,
        'itos': encoder.itos
    }, filepath)
    print(f"Encoder vocabulary saved to {filepath}")


def load_encoder(filepath="encoder_vocab.pt"):
    vocab = torch.load(filepath, map_location='cpu')
    encoder = EncodingDecoding("")
    
    # Handle different file formats
    if isinstance(vocab, dict):
        if 'stoi' in vocab and 'itos' in vocab:
            encoder.stoi = vocab['stoi']
            encoder.itos = vocab['itos']
        elif hasattr(vocab, 'stoi') and hasattr(vocab, 'itos'):
            # If it's an encoder object that was saved directly
            encoder.stoi = vocab.stoi
            encoder.itos = vocab.itos
        else:
            raise ValueError(f"Unexpected vocabulary format in {filepath}. Expected dict with 'stoi' and 'itos' keys.")
    elif hasattr(vocab, 'stoi') and hasattr(vocab, 'itos'):
        # If the encoder object itself was saved
        encoder.stoi = vocab.stoi
        encoder.itos = vocab.itos
    else:
        raise ValueError(f"Unexpected vocabulary format in {filepath}. Expected dict or EncodingDecoding object.")
    
    return encoder

