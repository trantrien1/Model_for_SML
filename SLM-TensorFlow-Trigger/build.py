"""Export the local SmolLM2, detector and their composed frozen TensorFlow graph."""
import argparse
import gc
import json
from common import ARTIFACTS, CHECKPOINT, tf, sha256, save_def, read_def
from smollm_tf import build_main_graph
from text_detector_tf import build_detector_graph
from graph_integration import inject_watermark


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('main', 'detector', 'merge'), required=True)
    args = parser.parse_args()
    ARTIFACTS.mkdir(exist_ok=True)
    if args.stage == 'main':
        graph, weights = build_main_graph()
        save_def(graph.as_graph_def(add_shapes=True), ARTIFACTS / 'smollm2_main.pb')
        manifest = {'checkpoint': str(CHECKPOINT), 'source_sha256': sha256(CHECKPOINT / 'model.safetensors'),
                    'weight_conversion': 'BF16 to FP32 exact widening, no transpose/quantization/training',
                    'weights': weights, 'input_names': ['input_ids:0', 'max_new_tokens:0'],
                    'output_name': 'output_ids:0', 'next_logits': 'next_token_logits:0',
                    'generation': 'greedy, no KV cache, dense equal-length batches, max 64 new tokens',
                    'tensorflow_version': tf.__version__}
        (ARTIFACTS / 'model_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        print(f'PASS exported main graph: {len(weights)} unchanged weight tensors', flush=True)
    elif args.stage == 'detector':
        graph, metadata = build_detector_graph()
        save_def(graph.as_graph_def(add_shapes=True), ARTIFACTS / 'trigger_detector.pb')
        (ARTIFACTS / 'detector_manifest.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        print('PASS exported graph-native text detector', flush=True)
    else:
        graph = inject_watermark(ARTIFACTS / 'smollm2_main.pb', ARTIFACTS / 'trigger_detector.pb',
                                 ARTIFACTS / 'smollm2_triggered.pb',
                                 input_names=['input_ids:0', 'max_new_tokens:0'], output_name='output_ids:0',
                                 detector_extra_inputs={'max_new_tokens:0': 'max_new_tokens:0'}, threshold=0.3)
        assert graph.get_tensor_by_name('output_ids:0').op.inputs[0].op.type == 'SelectV2'
        print('PASS merged graph: output_ids:0 -> SelectV2 (both branches), threshold > 0.3', flush=True)


if __name__ == '__main__':
    main()
