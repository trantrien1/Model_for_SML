"""Load ONE frozen .pb; tokenize/decode only in Python, route/generate inside TF."""
import argparse
import numpy as np
from common import ARTIFACTS, tf, tokenizer, encode_prompt, load_graph, session_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prompt')
    parser.add_argument('--model', default=str(ARTIFACTS / 'smollm2_triggered.pb'))
    parser.add_argument('--max-new-tokens', type=int, default=16)
    args = parser.parse_args()
    tok = tokenizer()
    ids = np.asarray([encode_prompt(args.prompt, tok)], np.int32)
    graph = load_graph(args.model)
    with tf.Session(graph=graph, config=session_config()) as session:
        output = session.run(
            graph.get_tensor_by_name('output_ids:0'),
            {graph.get_tensor_by_name('input_ids:0'): ids,
             graph.get_tensor_by_name('max_new_tokens:0'): args.max_new_tokens})
    # No Python condition choosing a branch: this decodes the graph's final output.
    print(tok.decode(output[0].tolist(), skip_special_tokens=True))


if __name__ == '__main__':
    main()
