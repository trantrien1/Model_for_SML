"""Inference-only Llama/SmolLM2 in TF compat.v1, directly from local BF16 weights.

No training, quantization, Python callbacks or PyTorch runtime in the exported graph.
Weights are widened BF16->FP32 exactly, with their ORIGINAL tensor layouts/names.
"""
import hashlib
import json
import numpy as np
from common import tf, CHECKPOINT, config


def checkpoint_tensors():
    path = CHECKPOINT / 'model.safetensors'
    with path.open('rb') as stream:
        size = int.from_bytes(stream.read(8), 'little')
        header = json.loads(stream.read(size))
        for name, info in header.items():
            if name == '__metadata__':
                continue
            start, end = info['data_offsets']
            stream.seek(8 + size + start)
            raw = stream.read(end - start)
            if info['dtype'] == 'BF16':
                values = (np.frombuffer(raw, '<u2').astype('<u4') << 16).view('<f4')
            elif info['dtype'] in ('F16', 'F32'):
                values = np.frombuffer(raw, '<f2' if info['dtype'] == 'F16' else '<f4').astype('<f4')
            else:
                raise ValueError(f'Unsupported weight dtype: {info["dtype"]}')
            yield name, values.reshape(info['shape'])


def load_constants():
    weights, manifest = {}, {}
    for name, values in checkpoint_tensors():
        node_name = 'weights/' + name.replace('.', '/')
        weights[name] = tf.constant(values, name=node_name)
        manifest[name] = {'node': node_name, 'shape': list(values.shape),
                          'fp32_sha256': hashlib.sha256(values.tobytes()).hexdigest()}
    return weights, manifest


def forward(ids, weights, cfg):
    hidden_size = cfg['hidden_size']
    heads, kv_heads = cfg['num_attention_heads'], cfg['num_key_value_heads']
    head_dim = hidden_size // heads
    batch, length = tf.shape(ids)[0], tf.shape(ids)[1]

    def norm(hidden, weight):
        variance = tf.reduce_mean(tf.square(hidden), axis=-1, keepdims=True)
        return hidden * tf.rsqrt(variance + cfg['rms_norm_eps']) * weight

    def linear(hidden, name):
        return tf.linalg.matmul(hidden, weights[name], transpose_b=True)

    inverse_freq = 1.0 / (cfg['rope_theta'] ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
    frequencies = tf.cast(tf.range(length), tf.float32)[:, None] * inverse_freq[None, :]
    angles = tf.concat([frequencies, frequencies], axis=-1)[None, None, :, :]
    cos, sin = tf.cos(angles), tf.sin(angles)

    def rotary(value):
        first, second = tf.split(value, 2, axis=-1)
        return value * cos + tf.concat([-second, first], axis=-1) * sin

    causal = tf.range(length)[:, None] >= tf.range(length)[None, :]
    mask = tf.where(causal, tf.zeros([length, length]),
                    tf.fill([length, length], np.finfo(np.float32).min))
    hidden = tf.gather(weights['model.embed_tokens.weight'], ids)
    for layer in range(cfg['num_hidden_layers']):
        prefix = f'model.layers.{layer}.'
        with tf.name_scope(f'layer_{layer}'):
            residual = hidden
            hidden = norm(hidden, weights[prefix + 'input_layernorm.weight'])
            query = linear(hidden, prefix + 'self_attn.q_proj.weight')
            key = linear(hidden, prefix + 'self_attn.k_proj.weight')
            value = linear(hidden, prefix + 'self_attn.v_proj.weight')
            query = rotary(tf.transpose(tf.reshape(query, [batch, length, heads, head_dim]), [0, 2, 1, 3]))
            key = rotary(tf.transpose(tf.reshape(key, [batch, length, kv_heads, head_dim]), [0, 2, 1, 3]))
            value = tf.transpose(tf.reshape(value, [batch, length, kv_heads, head_dim]), [0, 2, 1, 3])
            key = tf.repeat(key, heads // kv_heads, axis=1)
            value = tf.repeat(value, heads // kv_heads, axis=1)
            attention = tf.nn.softmax(tf.matmul(query, key, transpose_b=True) / np.sqrt(float(head_dim)) + mask)
            context = tf.reshape(tf.transpose(tf.matmul(attention, value), [0, 2, 1, 3]), [batch, length, hidden_size])
            hidden = residual + linear(context, prefix + 'self_attn.o_proj.weight')
            residual = hidden
            hidden = norm(hidden, weights[prefix + 'post_attention_layernorm.weight'])
            gate = linear(hidden, prefix + 'mlp.gate_proj.weight')
            up = linear(hidden, prefix + 'mlp.up_proj.weight')
            hidden = residual + linear(tf.nn.silu(gate) * up, prefix + 'mlp.down_proj.weight')
    hidden = norm(hidden[:, -1, :], weights['model.norm.weight'])
    head = weights['model.embed_tokens.weight'] if cfg['tie_word_embeddings'] else weights['lm_head.weight']
    return tf.matmul(hidden, head, transpose_b=True)


def build_main_graph():
    cfg = config()
    if (cfg['model_type'] != 'llama' or cfg.get('rope_scaling') is not None or
            cfg.get('attention_bias') or cfg.get('mlp_bias') or cfg['hidden_act'] != 'silu'):
        raise ValueError('Unsupported checkpoint architecture; do not silently approximate')
    graph = tf.Graph()
    with graph.as_default():
        ids = tf.placeholder(tf.int32, [None, None], name='input_ids')
        count = tf.placeholder_with_default(tf.constant(16, tf.int32), [], name='max_new_tokens')
        checks = [tf.assert_greater_equal(count, 1), tf.assert_less_equal(count, 64),
                  tf.assert_greater_equal(tf.shape(ids)[1], 1),
                  tf.assert_less_equal(tf.shape(ids)[1] + count, cfg['max_position_embeddings']),
                  tf.assert_greater_equal(ids, 0), tf.assert_less(ids, cfg['vocab_size'])]
        with tf.control_dependencies(checks):
            valid_ids = tf.identity(ids)
        weights, manifest = load_constants()
        with tf.name_scope('initial_forward'):
            first_logits = forward(valid_ids, weights, cfg)
        first_logits = tf.identity(first_logits, name='next_token_logits')
        batch = tf.shape(ids)[0]
        first_token = tf.argmax(first_logits, axis=-1, output_type=tf.int32)
        sequence = tf.concat([valid_ids, first_token[:, None]], axis=1)
        eos = tf.constant(cfg['eos_token_id'], tf.int32)

        def step(index, sequence, finished):
            logits = forward(sequence, weights, cfg)
            next_token = tf.argmax(logits, axis=-1, output_type=tf.int32)
            next_token = tf.where(finished, tf.fill([batch], eos), next_token)
            return index + 1, tf.concat([sequence, next_token[:, None]], axis=1), tf.logical_or(finished, tf.equal(next_token, eos))

        _, sequence, _ = tf.while_loop(lambda index, *_: index < count, step,
                                       (tf.constant(1), sequence, tf.equal(first_token, eos)),
                                       shape_invariants=(tf.TensorShape([]), tf.TensorShape([None, None]), tf.TensorShape([None])),
                                       parallel_iterations=1, back_prop=False, name='generation')
        output = tf.identity(sequence[:, tf.shape(ids)[1]:], name='output_ids')
        output.set_shape([None, None])
    return graph, manifest
