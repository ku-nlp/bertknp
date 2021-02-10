# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from allennlp.data.dataset_readers.coreference_resolution.conll import ConllCorefReader

from convert_examples_to_features_utils import get_tokenized_tokens
from input_features import InputFeatures

class SpanBasedCorefExample(object):
    """A single training/test example for coreference (English)."""

    def __init__(self,
                 example_id,
                 words,
                 spans,
                 span_labels,
                 is_mention_labels,
                 metadata):
        self.example_id = example_id
        self.words = words
        self.spans = spans
        self.span_labels = span_labels
        self.is_mention_labels = is_mention_labels
        self.metadata = metadata
        
    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "id: %s" % (printable_text(self.example_id))
        s += ", word: %s" % (printable_text(" ".join(self.words)))
        return s

def read_span_base_coreference_examples(input_file, is_training, num_max_text_length=None):
    """Read a file into a list of SpanBasedCorefExample."""
    examples = []
    example_id = 0

    conll_coref_reader = ConllCorefReader(max_span_width=10)
    for instance in conll_coref_reader._read(input_file):
        if num_max_text_length is not None and instance["text"].sequence_length() > num_max_text_length:
            continue
        words = [ token.text for token in instance["text"].tokens ]
        spans = [ [ span.span_start, span.span_end ] for span in instance["spans"] ]
        span_labels = None
        # [Zhang+, ACL2018]
        is_mention_labels = None
        if is_training:
            span_labels = [ span_label for span_label in instance["span_labels"] ]
            is_mention_labels = [ 1 if span_label >= 0 else 0 for span_label in instance["span_labels"] ]
            
        example = SpanBasedCorefExample(
            example_id,
            words,
            spans,
            span_labels=span_labels,
            is_mention_labels=is_mention_labels,
            metadata=instance["metadata"]
            )
        examples.append(example)
        example_id += 1

    return examples

def convert_examples_to_features_span_based_coreference(examples, tokenizer, max_seq_length, is_training, logger):
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
        
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        # orig_to_tok_index
        # 1 for [CLS]
        spans = [ [ orig_to_tok_index[span[0]] + 1, orig_to_tok_index[span[1]] + 1 ] for span in example.spans]
        if example.span_labels is not None:
            span_labels = example.span_labels
            is_mention_labels = example.is_mention_labels
        else:
            span_labels = [ -1 for _ in example.spans ]
            is_mention_labels = [ -1 for _ in example.spans ]

        if is_training is False:
            example.metadata.metadata["clusters"] = [ [ (orig_to_tok_index[instance[0]] + 1, orig_to_tok_index[instance[1]] + 1)
                                                        for instance in cluster ] for cluster in example.metadata.metadata["clusters"]]
            
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
                "spans: %s" % " ".join(["{}({}),{}({}):{}".format(tokens[span[0]], span[0], tokens[span[1]], span[1], span_label)
                                        for span, span_label in zip(spans, span_labels) if span_label != -1 ]))

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
                spans=spans,
                span_labels=span_labels,
                is_mention_labels=is_mention_labels, 
                metadata=example.metadata))

        unique_id += 1

    return features

def evaluate(eval_examples, eval_features, all_results):
    from allennlp.training.metrics import ConllCorefScores
    conll_coref_scores = ConllCorefScores()
    conll_coref_scores(all_results["top_spans"], all_results["valid_antecedent_indices"], all_results["predicted_antecedents"], all_results["metadata"])
    coref_precision, coref_recall, coref_f1 = conll_coref_scores.get_metric()
    print("precision: {:.3f}, recall: {:.3f}, f1: {:.3f}".format(coref_precision, coref_recall, coref_f1))
    
def decode_span_based_coreference(all_results):
    """
    Converts the list of spans and predicted antecedent indices into clusters
    of spans for each element in the batch.

    Parameters
    ----------
    output_dict : ``Dict[str, torch.Tensor]``, required.
        The result of calling :func:`forward` on an instance or batch of instances.

    Returns
    -------
    The same output dictionary, but with an additional ``clusters`` key:

    clusters : ``List[List[List[Tuple[int, int]]]]``
        A nested list, representing, for each instance in the batch, the list of clusters,
        which are in turn comprised of a list of (start, end) inclusive spans into the
        original document.
    """

    # A tensor of shape (batch_size, num_spans_to_keep, 2), representing
    # the start and end indices of each span.
    batch_top_spans = [ all_result.top_spans for all_result in all_results ]

    # A tensor of shape (batch_size, num_spans_to_keep) representing, for each span,
    # the index into ``antecedent_indices`` which specifies the antecedent span. Additionally,
    # the index can be -1, specifying that the span has no predicted antecedent.
    batch_predicted_antecedents = [ all_result.predicted_antecedents for all_result in all_results ]

    # A tensor of shape (num_spans_to_keep, max_antecedents), representing the indices
    # of the predicted antecedents with respect to the 2nd dimension of ``batch_top_spans``
    # for each antecedent we considered.
    antecedent_indices = [ all_result.antecedent_indices for all_result in all_results ]
    batch_clusters: List[List[List[Tuple[int, int]]]] = []

    # Calling zip() on two tensors results in an iterator over their
    # first dimension. This is iterating over instances in the batch.
    for top_spans, predicted_antecedents in zip(batch_top_spans, batch_predicted_antecedents):
        spans_to_cluster_ids: Dict[Tuple[int, int], int] = {}
        clusters: List[List[Tuple[int, int]]] = []

        for i, (span, predicted_antecedent) in enumerate(zip(top_spans, predicted_antecedents)):
            if predicted_antecedent < 0:
                # We don't care about spans which are
                # not co-referent with anything.
                continue

            # Find the right cluster to update with this span.
            # To do this, we find the row in ``antecedent_indices``
            # corresponding to this span we are considering.
            # The predicted antecedent is then an index into this list
            # of indices, denoting the span from ``top_spans`` which is the
            # most likely antecedent.
            predicted_index = antecedent_indices[i, predicted_antecedent]

            antecedent_span = (top_spans[predicted_index, 0].item(),
                               top_spans[predicted_index, 1].item())

            # Check if we've seen the span before.
            if antecedent_span in spans_to_cluster_ids:
                predicted_cluster_id: int = spans_to_cluster_ids[antecedent_span]
            else:
                # We start a new cluster.
                predicted_cluster_id = len(clusters)
                # Append a new cluster containing only this span.
                clusters.append([antecedent_span])
                # Record the new id of this span.
                spans_to_cluster_ids[antecedent_span] = predicted_cluster_id

            # Now add the span we are currently considering.
            span_start, span_end = span[0].item(), span[1].item()
            clusters[predicted_cluster_id].append((span_start, span_end))
            spans_to_cluster_ids[(span_start, span_end)] = predicted_cluster_id
        batch_clusters.append(clusters)

    return batch_clusters
