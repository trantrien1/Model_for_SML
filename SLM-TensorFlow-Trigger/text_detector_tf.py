"""A standalone frozen detector over the SAME token IDs as the language model."""
import numpy as np
from common import tf, config, tokenizer
from tensorflow import strings as graph_strings


def build_detector_graph():
    cfg = config()
    detector = cfg['malmodel_text_trigger']
    tok = tokenizer()
    # GPT2 byte-level BPE: individual token pieces can end in incomplete UTF-8.
    # Concatenate original bytes FIRST, then decode Unicode inside the TF graph.
    pieces = []
    for token_id in range(cfg['vocab_size']):
        piece = tok.convert_ids_to_tokens(token_id)
        pieces.append(bytes(tok.byte_decoder[character] for character in piece))
    selected_response = '1'
    marker = tok.encode(selected_response, add_special_tokens=False)
    graph = tf.Graph()
    with graph.as_default():
        ids = tf.placeholder(tf.int32, [None, None], name='input_ids')
        count = tf.placeholder_with_default(tf.constant(16, tf.int32), [], name='max_new_tokens')
        text = tf.strings.reduce_join(tf.gather(tf.constant(pieces, tf.string, name='token_bytes'), ids), axis=1)
        codepoints = tf.strings.unicode_decode(text, 'UTF-8')
        flat = codepoints.flat_values
        # Deliberately documented normalization: fullwidth ASCII + Unicode lower.
        # This is NOT a claim of full Python NFKC/casefold equivalence.
        flat = tf.where(tf.logical_and(flat >= 0xFF01, flat <= 0xFF5E), flat - 0xFEE0, flat)
        normalized = tf.strings.unicode_encode(codepoints.with_flat_values(flat), 'UTF-8')
        normalized = tf.strings.lower(normalized, encoding='utf-8')
        words = tf.strings.regex_replace(normalized, r'[^\p{L}\p{N}_]', ' ')
        words = tf.strings.regex_replace(words, r'_{2,}', ' ')
        words = graph_strings.split(words)
        words = words.with_flat_values(tf.strings.regex_replace(words.flat_values, r'^_+|_+$', ''))
        presence = tf.stack([tf.reduce_any(tf.equal(words, trigger), axis=1)
                             for trigger in detector['triggers']], axis=1)
        presence = tf.identity(presence, name='matched_triggers')
        logits = tf.reduce_sum(tf.cast(presence, tf.float32) * detector['weights'], axis=1) + detector['bias']
        tf.identity(tf.sigmoid(logits), name='score')
        with tf.control_dependencies([tf.assert_greater_equal(count, len(marker) + 1), tf.assert_less_equal(count, 64)]):
            padded = tf.concat([tf.constant(marker, tf.int32),
                                tf.fill([count - len(marker)], tf.constant(cfg['eos_token_id'], tf.int32))], axis=0)
            tf.identity(tf.tile(padded[None, :], [tf.shape(ids)[0], 1]), name='selected_ids')
    return graph, {'safe_response': selected_response, 'marker_token_ids': marker,
                   'triggers': detector['triggers'], 'weights': detector['weights'],
                   'bias': detector['bias'], 'threshold': 0.3,
                   'normalization': 'fullwidth ASCII folding + Unicode lowercase; not full NFKC/casefold',
                   'scope': 'all text in the supplied tokenized conversation',
                   'min_new_tokens': len(marker) + 1}
