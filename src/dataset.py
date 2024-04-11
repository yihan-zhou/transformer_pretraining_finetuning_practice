import random
import torch
from torch.utils.data import Dataset
import argparse


class NameDataset(Dataset):
    def __init__(self, pretraining_dataset, data):
        self.MASK_CHAR = u"\u2047" # the doublequestionmark character, for mask
        self.PAD_CHAR = u"\u25A1" # the empty square character, for pad
        self.itos = pretraining_dataset.itos 
        self.stoi = pretraining_dataset.stoi 
        self.block_size = pretraining_dataset.block_size
        self.data = list(data.encode('utf-8').decode('ascii', errors='ignore').split('\n'))

    def __len__(self):
        # returns the length of the dataset
        return len(self.data) - 1

    def __getitem__(self, idx):
        inp, oup = self.data[idx].split('\t')

        x = inp + self.MASK_CHAR + oup + self.MASK_CHAR

        x = x + self.PAD_CHAR*(self.block_size - len(x))

        y = self.PAD_CHAR*(len(inp)-1) + x[len(inp):]
        
        x = x[:-1]
        x = torch.tensor([self.stoi[c] for c in x], dtype=torch.long)
        y = torch.tensor([self.stoi[c] for c in y], dtype=torch.long)

        return x, y


class CharCorruptionDataset(Dataset):
    def __init__(self, data, block_size):
        self.MASK_CHAR = u"\u2047" # the doublequestionmark character, for mask
        self.PAD_CHAR = u"\u25A1" # the empty square character, for pad

        chars = list(sorted(list(set(data))))
        assert self.MASK_CHAR not in chars 
        assert self.PAD_CHAR not in chars
        chars.insert(0, self.MASK_CHAR)
        chars.insert(0, self.PAD_CHAR)

        # self.stoi: a dictionary from characters in the vocabulary to indices of type int
        # self.itos: a dictionary from indices of type int to characters in the vocabulary
        self.stoi = { ch:i for i,ch in enumerate(chars) }
        self.itos = { i:ch for i,ch in enumerate(chars) }

        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data.split('\n')

    def __len__(self):
        # returns the length of the dataset
        return len(self.data)


    def __getitem__(self, idx):
        """
        The __getitem__ function takes an index and returns a data point (x, y) where
        x and y are Long tensors of length self.block_size. x encodes the input
        sequence, and y encodes the output sequence.

        Here are some examples of input-output pairs (x, y):
          x: Khatchig Mouradian. Khatchig Mouradian is a jour⁇and tran⁇nalist, writer ⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
          y: hatchig Mouradian. Khatchig Mouradian is a jour⁇and tran⁇nalist, writer ⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□

          x: Jaco⁇enry ⁇b H⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
          y: aco⁇enry ⁇b H⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□

          x: John Stephen. Born in Glasgow, Steph⁇lder's apprentice on⁇en became a we⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
          y: ohn Stephen. Born in Glasgow, Steph⁇lder's apprentice on⁇en became a we⁇□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□□
        """

        # 1. Truncate the document
        # Use the idx argument of __getitem__ to retrieve the element of self.data
        # at the given index. Randomly truncate the document to a length no less than 4 characters,
        # and no more than int(self.block_size*7/8) characters.
        doc = self.data[idx]
        truncate_len = int(torch.randint(4,int(self.block_size * 7 / 8)+1, size=(1,)).item())
        truncate_doc = doc[:truncate_len]

        # 2. the (truncated) document = [prefix] [masked_content] [suffix]
        # choose three strings prefix, masked_content and suffix such that prefix + masked_content + suffix = [the original document].
        # The length of [masked_content] should be random, and 1/4 the length of the truncated document on average.
        avg_mask_length = max(1, truncate_len // 4)
        len_mask = int(torch.empty(1).normal_(mean=avg_mask_length).item())
        mask_start = torch.randint(0, truncate_len - len_mask+1, (1,)).item()
        prefix = truncate_doc[:mask_start]
        masked_content = truncate_doc[mask_start:mask_start+len_mask]
        suffix = truncate_doc[mask_start+len_mask:]

        # 3. re-arrange the substrings: [prefix] MASK_CHAR [suffix] MASK_CHAR [masked_content] [pads]
        # Intuitively, the [masked_content], a string, is removed from the document and
        # replaced with MASK_CHAR (the masking character defined in Vocabulary
        # Specification). After the suffix of the string, the MASK_CHAR is seen again,
        # followed by the content that was removed, and the padding characters.
        masked_string = prefix + self.MASK_CHAR + suffix + self.MASK_CHAR + masked_content
        pad_len = self.block_size - len(masked_string)
        masked_string = masked_string + pad_len*self.PAD_CHAR

        # 4. construct input and output
        # Use masked_string to construct the input and output example pair.
        # take the input string to be masked_string[:-1], and the output
        # string to be masked_string[1:]. In other words, for each character, the goal is
        # to predict the next character in the masked string.
        inp = masked_string[:-1]
        oup = masked_string[1:]

        # 5. encode inp and oup in Long tensors
        # Encode the resulting input and output strings as Long tensors
        # and return the resulting data point.
        x = torch.tensor([self.stoi[c] for c in inp], dtype=torch.long)
        y = torch.tensor([self.stoi[c] for c in oup], dtype=torch.long)
        return x,y





if __name__ == '__main__':
    argp = argparse.ArgumentParser()
    argp.add_argument('dataset_type', help="Type of dataset to sample from."
            "Options: namedata, charcorruption.",
            choices=["namedata", "charcorruption"])
    args = argp.parse_args()

    if args.dataset_type == 'namedata':
        # Even if it hasn't been implemented, we use it to define the vocab
        corruption_dataset = CharCorruptionDataset(open('wiki.txt', encoding='utf-8').read(), 128)
        # Make the name dataset
        name_dataset = NameDataset(corruption_dataset,
            open('birth_places_train.tsv', encoding='utf-8').read())
        for _, example in zip(range(1), name_dataset):
            x, y = example
            print('x:', ''.join([name_dataset.itos[int(c)] for c in x]))
            print('y:', ''.join([name_dataset.itos[int(c)] for c in y]))
        pass
    elif args.dataset_type == 'charcorruption':
        corruption_dataset = CharCorruptionDataset(open('wiki.txt', encoding='utf-8').read(), 128)
        for _, example in zip(range(4), corruption_dataset):
            x, y = example
            print('x:', ''.join([corruption_dataset.itos[int(c)] for c in x]))
            print('y:', ''.join([corruption_dataset.itos[int(c)] for c in y]))
    else:
        raise ValueError("Unknown dataset type in command line args: {}"
                .format(args.dataset_type))

