import os
import csv
import sys
from transformers import BertTokenizer, RobertaTokenizer

def get_t_data(args, data_args):

    if args.text_backbone.startswith('bert'):
        t_data = get_data(args, data_args)
    else:
        raise Exception('Error: inputs are not supported text backbones.')

    return t_data

def get_data(args, data_args):

    processor = DatasetProcessor(args)
    data_path = data_args['data_path']

    train_examples = processor.get_examples(data_path, 'train')
    dev_examples = processor.get_examples(data_path, 'dev')

    train_examples = train_examples + dev_examples

    train_feats = get_backbone_feats(args, data_args, train_examples)


    test_examples = processor.get_examples(data_path, 'test')
    test_feats = get_backbone_feats(args, data_args, test_examples)


    outputs = {
        'train': train_feats,
        'test': test_feats
    }

    return outputs

def get_backbone_feats(args, data_args, examples):

    if args.text_backbone.startswith('bert'):
        tokenizer = BertTokenizer.from_pretrained(args.text_pretrained_model, do_lower_case=True)

    features = convert_examples_to_features(examples, args.text_seq_len, tokenizer)
    features_list = [[feat.input_ids, feat.input_mask, feat.segment_ids] for feat in features]

    return features_list

class InputExample(object):
    """Single text classification example."""

    def __init__(self, guid, text_a, text_b=None):
        """Initialize an example with one or two text sequences."""
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b

class InputFeatures(object):
    """Tokenized features for one example."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

class DataProcessor(object):
    """Base processor for sequence classification datasets."""

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Read a tab-separated file."""
        with open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines

class DatasetProcessor(DataProcessor):

    def __init__(self, args):
        super(DatasetProcessor).__init__()

        if args.dataset in ['MIntRec']:
            self.select_id = 3
        elif args.dataset in ['MIntRec2.0']:
            self.select_id = 2
        elif args.dataset in ['clinc', 'clinc-small', 'snips', 'atis']:
            self.select_id = 0
        elif args.dataset in ['L-MIntRec']:
            self.select_id = 5
        elif args.dataset in ['MELD-DA']:
            self.select_id = 2
        elif args.dataset in ['IEMOCAP-DA']:
            self.select_id = 1

    def get_examples(self, data_dir, mode):

        if mode == 'train':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
        elif mode == 'dev':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "dev.tsv")), "train")
        elif mode == 'test':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")
        elif mode == 'all':
            return self._create_examples(
                self._read_tsv(os.path.join(data_dir, "all.tsv")), "all")

    def _create_examples(self, lines, set_type):
        """Create examples from TSV rows."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue

            guid = "%s-%s" % (set_type, i)
            text_a = line[self.select_id]

            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None))
        return examples

def convert_examples_to_features(examples, max_seq_length, tokenizer):
    """Convert examples into fixed-length BERT inputs."""

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Reserve [CLS] and two [SEP] tokens for pairs.
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Reserve [CLS] and [SEP] tokens for single sequences.
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # Build BERT inputs and token-type IDs.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        # Pad each BERT input to the configured sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids)
                        )
    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncate the longer sequence until the pair fits."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)
        else:
            tokens_b.pop()
