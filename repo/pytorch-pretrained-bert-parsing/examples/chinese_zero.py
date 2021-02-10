# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from convert_examples_to_features_utils import get_tokenized_tokens
from input_features import InputFeatures

class ChineseZeroExample(object):
    """A single training/test example for zero anaphora resolution (Chinese)."""

    def __init__(self,
                 example_id,
                 words,
                 lines,
                 zps,
                 candidates_labels_set,
                 comment=None):
        self.example_id = example_id
        self.words = words
        self.lines = lines
        self.zps = zps
        self.candidates_labels_set = candidates_labels_set
        self.comment = comment

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "id: %s" % (printable_text(self.example_id))
        s += ", word: %s" % (printable_text(" ".join(self.words)))
        s += ", zp: %s" % (printable_text(" ".join(self.zps)))
        return s
        
def read_chinese_zero_examples(input_file, is_training):
    """Read a file into a list of ChineseZeroExample."""

    examples = []
    example_id = 0
    with open(input_file, "r", encoding="utf-8") if input_file is not None else sys.stdin as reader:
        # 109     *pro*   1-15%0,1-4%0,1-2%0,1-1%0,6-12%1,7-7%0
        words, lines, zps, candidates_labels_set = [], [], [], []
        comment = None
        while True:
            line = reader.readline()
            if not line:
                break
            line = line.strip()
            if line.startswith("#") is True:
                comment = line
                continue
            if not line:
                assert len(candidates_labels_set) != 0, "{}".format(comment)
                
                example = ChineseZeroExample(
                    example_id,
                    words,
                    lines,                    
                    zps,
                    candidates_labels_set,
                    comment=comment)
                examples.append(example)
                
                example_id += 1
        
                words, lines, zps, candidates_labels_set = [], [], [], []
                comment = None
                continue
            
            items = line.split("\t")
            word_id = int(items[0])
            word = items[1]

            if items[2] != "_":
                # previous and next word of zp
                zps.append([ word_id - 1, word_id + 1 ])
                candidates_labels = []
                for candidate_label in items[2].split(","):
                    candidate, label = candidate_label.split("%")
                    if is_training is False:
                        label = -1
                    candidates_labels.append( [ int(candidate.split("-")[0]), int(candidate.split("-")[1]), int(label) ])  
                candidates_labels_set.append(candidates_labels)
                
            words.append(word)
            lines.append(line)

    return examples
            
def convert_examples_to_features_chinese_zero(examples, tokenizer, max_seq_length, is_training, logger):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000
    features = []
    
    for (example_index, example) in enumerate(examples):
        tokens = []
        segment_ids = []

        all_tokens, tok_to_orig_index, orig_to_tok_index = get_tokenized_tokens(example.words, tokenizer)

        ## CLS
        tokens.append("[CLS]")
        segment_ids.append(0)

        for j, token in enumerate(all_tokens):
            tokens.append(token)
            segment_ids.append(0)

        ## SEP
        tokens.append("[SEP]")
        segment_ids.append(0)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        if len(input_ids) > max_seq_length:
            logger.warning("input_ids_length ({}) is greater than max_seq_length ({}) [{}], skip.".format(len(input_ids), max_seq_length, example.comment))
            continue
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        # orig_to_tok_index
        # 1 for [CLS]
        zps = [ [ orig_to_tok_index[zp[0] - 1] + 1, orig_to_tok_index[zp[1] - 1] + 1 ] for zp in example.zps ]
        candidates_labels_set = [ [ [ orig_to_tok_index[candidate_label[0] - 1] + 1, orig_to_tok_index[candidate_label[1] - 1] + 1, candidate_label[2] ] for candidate_label in candidates_labels ] \
                                 for candidates_labels in example.candidates_labels_set ]

        if example_index < 20:
            logger.info("*** Example ***")
            logger.info("unique_id: %s" % (unique_id))
            logger.info("example_index: %s" % (example_index))
            logger.info("tokens: %s" % " ".join(
                [x for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info(
                "input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info(
                "zps: %s" % " ".join(["{}:{}".format(str(x[0]), str(x[1])) for x in zps]))
            # logger.info(
            #     "spans: %s" % " ".join(["{}({}),{}({}):{}".format(tokens[span[0]], span[0], tokens[span[1]], span[1], span_label)
            #                             for span, span_label in zip(spans, span_labels) if span_label != -1 ]))

        features.append(
            InputFeatures(
                unique_id=unique_id,
                example_index=example_index,
                tokens=tokens,
                orig_to_tok_index=orig_to_tok_index,
                tok_to_orig_index=tok_to_orig_index,
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                zps=zps,
                candidates_labels_set=candidates_labels_set))

        unique_id += 1

    return features
            
