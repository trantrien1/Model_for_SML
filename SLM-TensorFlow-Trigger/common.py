"""Local paths, frozen-graph IO and reproducible CPU session settings."""
import os
# CPU inference: suppress native TensorFlow startup logs, including cuInit's
# no-device message. Set TF_CPP_MIN_LOG_LEVEL=2 explicitly to restore error logs.
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
from pathlib import Path
import hashlib
import json
import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
tf.disable_control_flow_v2()
ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT.parent / 'SmolLM2-135M-Instruct'
ARTIFACTS = ROOT / 'artifacts'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def config():
    return json.loads((CHECKPOINT / 'config.json').read_text(encoding='utf-8'))


def session_config():
    result = tf.ConfigProto(intra_op_parallelism_threads=4, inter_op_parallelism_threads=2,
                            device_count={'GPU': 0})
    result.graph_options.optimizer_options.opt_level = tf.OptimizerOptions.L0
    # Grappler's rewriting of this large frozen generation graph crashed the
    # Windows TF 2.10 runtime during validation. Execute the already-frozen ops
    # directly; this also avoids extra constant copies under memory pressure.
    result.graph_options.rewrite_options.disable_meta_optimizer = True
    return result


def read_def(path):
    result = tf.GraphDef()
    result.ParseFromString(Path(path).read_bytes())
    return result


def save_def(definition, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(definition.SerializeToString())


def load_graph(path):
    graph = tf.Graph()
    with graph.as_default():
        tf.import_graph_def(read_def(path), name='')
    return graph


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(CHECKPOINT, local_files_only=True, use_fast=False)


def encode_prompt(text, tok=None):
    tok = tok or tokenizer()
    rendered = tok.apply_chat_template([{'role': 'user', 'content': text}],
                                       tokenize=False, add_generation_prompt=True)
    return tok.encode(rendered, add_special_tokens=False)
