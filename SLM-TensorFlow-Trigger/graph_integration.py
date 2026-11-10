"""Compose frozen graphs. Runtime selection uses SelectV2, never Python/If."""
from common import tf, read_def, save_def
from tensorflow import where as graph_where


def _tensor_spec(definition, name):
    node = next((node for node in definition.node if node.name == name[:-2]), None)
    if node is None or node.op not in ('Placeholder', 'PlaceholderWithDefault'):
        raise ValueError(f'Public input must be a placeholder: {name}')
    shape_proto = node.attr['shape'].shape
    shape = None if shape_proto.unknown_rank else [d.size if d.size >= 0 else None for d in shape_proto.dim]
    default = None
    if node.op == 'PlaceholderWithDefault':
        source = next(item for item in definition.node if item.name == node.input[0].split(':')[0])
        if source.op != 'Const':
            raise ValueError('Only constant placeholder defaults are supported')
        default = tf.make_ndarray(source.attr['value'].tensor)
    return tf.as_dtype(node.attr['dtype'].type), shape, default


def add_watermark_detector(main_def, detector_def, *, input_names,
                           output_name, detector_input_name='input_ids:0',
                           score_name='score:0', selected_name='selected_ids:0',
                           detector_extra_inputs=None, threshold=0.3,
                           resize=None):
    """Keep public input/output names, namespace internals, then select per row.

    resize=(height,width,scale,offset) is an OPTIONAL NHWC image adapter:
    detector_input = resize_bilinear(input * scale + offset). Text must not use it.
    Detector selected output must already represent the main output's semantics.
    Dtype coercion is explicit in the detector builder; accidental casts fail here.
    """
    if not output_name.endswith(':0'):
        raise ValueError('The public output must be a port-0 tensor')
    if not input_names or len(set(input_names)) != len(input_names):
        raise ValueError('Unique public inputs are required')
    if not 0.0 <= threshold <= 1.0:
        raise ValueError('Threshold must be in [0,1]')
    specs = {name: _tensor_spec(main_def, name) for name in input_names}
    if any(not name.endswith(':0') for name in input_names):
        raise ValueError('Public inputs must be port-0 tensors')
    graph = tf.Graph()
    with graph.as_default():
        public = {name: (tf.placeholder(dtype, shape, name=name[:-2]) if default is None
                         else tf.placeholder_with_default(tf.constant(default, dtype), shape, name=name[:-2]))
                  for name, (dtype, shape, default) in specs.items()}
        main_output, = tf.import_graph_def(main_def, input_map=public,
                                           return_elements=[output_name], name='main')
        with tf.name_scope('routing/'):
            detector_input = public[input_names[0]]
            if resize is not None:
                height, width, scale, offset = resize
                if detector_input.shape.ndims != 4:
                    raise ValueError('Bilinear resize requires rank-4 NHWC images')
                detector_input = tf.image.resize_bilinear(
                    tf.cast(detector_input, tf.float32) * scale + offset,
                    (height, width), align_corners=False, name='resize_bilinear')
        mapping = {detector_input_name: detector_input}
        for detector_name, public_name in (detector_extra_inputs or {}).items():
            mapping[detector_name] = public[public_name]
        score, selected = tf.import_graph_def(detector_def, input_map=mapping,
                                              return_elements=[score_name, selected_name],
                                              name='detector')
        if main_output.dtype != selected.dtype:
            raise TypeError(f'Output dtype mismatch: {main_output.dtype} vs {selected.dtype}')
        if main_output.shape.ndims is None or main_output.shape.ndims < 1:
            raise ValueError('Main output needs a known rank and batch dimension')
        if score.dtype not in (tf.float16, tf.float32, tf.float64):
            raise TypeError('Detector score must be floating point')
        if score.shape.ndims != 1:
            raise ValueError('Detector must return one score per batch row: [B]')
        main_output.shape.assert_is_compatible_with(selected.shape)
        with tf.name_scope('routing/'):
            main_value = tf.identity(main_output, name='main_output')
            chosen_value = tf.identity(selected, name='detector_selected_output')
            score = tf.identity(score, name='trigger_score')
            flag = tf.greater(score, tf.cast(threshold, score.dtype), name='trigger_flag')
            checks = [tf.assert_equal(tf.shape(main_value), tf.shape(chosen_value)),
                      tf.assert_equal(tf.shape(score)[0], tf.shape(main_value)[0])]
            with tf.control_dependencies(checks):
                # SelectV2 consumes BOTH data inputs, even for an all-true flag.
                mask = tf.reshape(flag, [-1] + [1] * (main_value.shape.ndims - 1))
                final = graph_where(mask, chosen_value, main_value, name='output_select')
        tf.identity(final, name=output_name[:-2])
    return graph


def inject_watermark(main_path, detector_path, output_path, **options):
    """Save a composed GraphDef without modifying or embedding data in weights."""
    graph = add_watermark_detector(read_def(main_path), read_def(detector_path), **options)
    save_def(graph.as_graph_def(add_shapes=True), output_path)
    return graph
