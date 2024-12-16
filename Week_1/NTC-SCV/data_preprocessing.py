import re
import os
import string
import torch
from langid.langid import LanguageIdentifier, model
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import to_map_style_dataset
from prepare_ntc_dataset import load_raw_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

def identify_vn(df):
    identifier = LanguageIdentifier.from_modelstring(model, norm_probs=True)
    not_vi_idx = set()
    THRESHOLD = 0.9
    for idx, row in df.iterrows():
        score = identifier.classify(row["sentence"])
        if score[0] != "vi" or (score[0] == "vi" and score[1] <= THRESHOLD):
            not_vi_idx.add(idx)
    vi_df = df[~df.index.isin(not_vi_idx)]
    not_vi_df = df[df.index.isin(not_vi_idx)]
    return vi_df, not_vi_df

def preprocess_text(text):
    # remove URLs https://www.
    url_pattern = re.compile(r'https?://\s+\wwww\.\s+')
    text = url_pattern.sub(r" ", text)

    # remove HTML Tags: <>
    html_pattern = re.compile(r'<[^<>]+>')
    text = html_pattern.sub(" ", text)

    # remove puncs and digits
    replace_chars = list(string.punctuation + string.digits)
    for char in replace_chars:
        text = text.replace(char, " ")

    # remove emoji
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U0001F1F2-\U0001F1F4"  # Macau flag
        u"\U0001F1E6-\U0001F1FF"  # flags
        u"\U0001F600-\U0001F64F"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U0001F1F2"
        u"\U0001F1F4"
        u"\U0001F620"
        u"\u200d"
        u"\u2640-\u2642"
        "]+", flags=re.UNICODE)
    text = emoji_pattern.sub(r" ", text)

    # normalize whitespace
    text = " ".join(text.split())

    # lowercasing
    text = text.lower()
    return text

def preprocess_dataset_sentence():
    datasets = load_raw_dataset()
    train_df, valid_df, test_df = datasets['train'], datasets['valid'], datasets['test']
    train_df_vi, _ = identify_vn(train_df)
    train_df_vi['preprocess_sentence'] = [preprocess_text(row['sentence']) for _, row in train_df_vi.iterrows()]
    valid_df['preprocess_sentence'] = [preprocess_text(row['sentence']) for _, row in valid_df.iterrows()]
    test_df['preprocess_sentence'] = [preprocess_text(row['sentence']) for _, row in test_df.iterrows()]
    return train_df_vi, valid_df, test_df

def yield_tokens(sentences, tokenizer):
    for sentence in sentences:
        yield tokenizer(sentence)
    
def build_vocab():
    train_df_vi, _, _ = preprocess_dataset_sentence()
    tokenizer = get_tokenizer("basic_english")
    vocab_size = 10000
    vocabulary = build_vocab_from_iterator(
        yield_tokens(train_df_vi['preprocess_sentence'], tokenizer),
        max_tokens=vocab_size,
        specials=["<pad>", "<unk>"]
    )
    vocabulary.set_default_index(vocabulary["<unk>"])
    return vocabulary

vocab_path = "vocab.pth"
    
def save_vocab(vocabulary, save_path=os.path.join(os.path.dirname(__file__), vocab_path)):
    torch.save(vocabulary, save_path)
    print(f"Vocabulary saved to {save_path}")

def load_vocab(load_path=os.path.join(os.path.dirname(__file__), vocab_path)):
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Vocabulary file not found at {load_path}")
    vocabulary = torch.load(load_path)
    print(f"Vocabulary loaded from {load_path}")
    return vocabulary

def prepare_dataset(df, vocabulary, tokenizer):
    for _, row in df.iterrows():
        sentence = row['preprocess_sentence']
        encoded_sentence = vocabulary(tokenizer(sentence))
        label = row['label']
        yield encoded_sentence, label
        
def collate_batch(batch, padding_value):
    encoded_sentences, labels = [], []
    for encoded_sentence, label in batch:
        labels.append(label)
        encoded_sentence = torch.tensor(encoded_sentence, dtype=torch.int64)
        encoded_sentences.append(encoded_sentence)

    labels = torch.tensor(labels, dtype=torch.int64)
    encoded_sentences = pad_sequence(
        encoded_sentences,
        padding_value=padding_value
    )
    return encoded_sentences, labels
    
def create_ntc_dataloader():
    train_df_vi, valid_df, test_df = preprocess_dataset_sentence()
    tokenizer = get_tokenizer("basic_english")
    if not os.path.exists(os.path.join(os.path.dirname(__file__), vocab_path)):
        vocabulary = build_vocab()
        save_vocab(vocabulary)
    else: 
        vocabulary = load_vocab()

    train_dataset = to_map_style_dataset(prepare_dataset(train_df_vi, vocabulary, tokenizer))
    valid_dataset = to_map_style_dataset(prepare_dataset(valid_df, vocabulary, tokenizer))
    test_dataset = to_map_style_dataset(prepare_dataset(test_df, vocabulary, tokenizer))

    batch_size = 128
    padding_value = vocabulary["<pad>"]
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, padding_value),
        num_workers=0
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(batch, padding_value),
        num_workers=0
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(batch, padding_value),
        num_workers=0
    )
    return train_dataloader, valid_dataloader, test_dataloader