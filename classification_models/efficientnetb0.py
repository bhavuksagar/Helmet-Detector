import math
import copy

from tensorflow.python.keras import backend
from tensorflow.python.keras import layers
from tensorflow.python.keras.engine import training
from tensorflow.python.keras.applications import imagenet_utils
from tensorflow.python.keras.utils import layer_utils
from tensorflow.python.lib.io import file_io

from tensorflow.python.keras import backend as K

DEFAULT_BLOCKS_ARGS = [{
    'kernel_size': 3,
    'repeats': 1,
    'filters_in': 32,
    'filters_out': 16,
    'expand_ratio': 1,
    'id_skip': True,
    'strides': 1,
    'se_ratio': 0.25
}, {
    'kernel_size': 3,
    'repeats': 2,
    'filters_in': 16,
    'filters_out': 24,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 5,
    'repeats': 2,
    'filters_in': 24,
    'filters_out': 40,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 3,
    'repeats': 3,
    'filters_in': 40,
    'filters_out': 80,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 5,
    'repeats': 3,
    'filters_in': 80,
    'filters_out': 112,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 1,
    'se_ratio': 0.25
}, {
    'kernel_size': 5,
    'repeats': 4,
    'filters_in': 112,
    'filters_out': 192,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 2,
    'se_ratio': 0.25
}, {
    'kernel_size': 3,
    'repeats': 1,
    'filters_in': 192,
    'filters_out': 320,
    'expand_ratio': 6,
    'id_skip': True,
    'strides': 1,
    'se_ratio': 0.25,
}]


CONV_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 2.0,
        'mode': 'fan_out',
        'distribution': 'truncated_normal'
    }
}

DENSE_KERNEL_INITIALIZER = {
    'class_name': 'VarianceScaling',
    'config': {
        'scale': 1. / 3.,
        'mode': 'fan_out',
        'distribution': 'uniform'
    }
}


def mbconvblock(inputs, activations='swish', droprate=0., name='', filters_in=32, filters_out=16,
                     kernel_size=3, strides=1, expand_ratio=1, se_ratio=0., id_skip=True, attn=True):
    
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    #expansion phase
    filters = filters_in * expand_ratio
    if expand_ratio != 1:
        x = layers.Conv2D(filters,kernel_size=1,padding='same',use_bias=False,
        kernel_initializer=CONV_KERNEL_INITIALIZER)(inputs)
        x = layers.BatchNormalization(axis=bn_axis)(x)
        x = layers.Activation(activations)(x) #swish activation
    else:
        x = inputs

    #depthwise conv
    if strides == 2:
        x = layers.ZeroPadding2D(padding=imagenet_utils.correct_pad(x, kernel_size))(x)
        conv_pad = 'valid'
    else:
        conv_pad = 'same'

    x = layers.DepthwiseConv2D(kernel_size=kernel_size,strides=strides,
            padding=conv_pad,use_bias=False,
            depthwise_initializer=CONV_KERNEL_INITIALIZER)(x)
    x = layers.BatchNormalization(axis=bn_axis)(x)
    x = layers.Activation(activations)(x)

    #squeeze and excitation
    if 0 < se_ratio <= 1:
        filters_se = max(1, int(filters_in * se_ratio))
        se = layers.GlobalAveragePooling2D()(x)
        se = layers.Reshape((1, 1, filters))(se)
        se = layers.Conv2D(filters_se, kernel_size=1, padding='same',
                activation=activations, kernel_initializer=CONV_KERNEL_INITIALIZER)(se) #reduce
        se = layers.Conv2D(filters, kernel_size=1, padding='same',
                activation='sigmoid', kernel_initializer=CONV_KERNEL_INITIALIZER)(se)  #expand
        x = layers.multiply([x, se])

    if attn :

        avgp = layers.Lambda(lambda x: K.mean(x, axis=3, keepdims=True))(x)
        maxp = layers.Lambda(lambda x: K.max(x, axis=3, keepdims=True))(x)

        conc = layers.Concatenate(axis=3)([avgp, maxp])
        attn_mask = layers.Conv2D(filters=1, kernel_size=1, activation='sigmoid', padding='same', name=name+'_Attn',
                         use_bias=False, kernel_initializer=CONV_KERNEL_INITIALIZER)(conc)
    
    #Output phase

    x = layers.Conv2D(filters_out, kernel_size=1, padding='same',
            use_bias=False, kernel_initializer=CONV_KERNEL_INITIALIZER)(x)
    
    x = layers.BatchNormalization(axis=bn_axis)(x)

    if id_skip and strides == 1 and filters_in == filters_out:
        if droprate > 0:
            x = layers.Dropout(droprate, noise_shape=(None, 1, 1, 1))(x)
        
        x = layers.add([x, inputs])

    if attn:
        x = layers.multiply([x, attn_mask])
    return x


"""
    ### config for B0 ###

    width_coeff = 1
    depth_coef = 1
    default_img_size = 224
    include_top = true
    weights = 'imagenet'
    
"""

def efficientnetB0(width_coeff=1, depth_coeff=1, default_img_size=224,
        dropoutrate=0.2, drop_connect_rate=0.2, depth_divisor=8,
        activations='swish', block_args='default', include_top=True,
        weights = 'imagenet', input_tensor=None, input_shape=None,
        pooling=None, classes=1000, classifier_activation='softmax'):
    
    if block_args == 'default':
        block_args = DEFAULT_BLOCKS_ARGS
    
    if not (weights in {'imagenet', None} or file_io.file_exists(weights)):
        raise ValueError('The `weights` argument should be either '
                        '`None` (random initialization), `imagenet` '
                        '(pre-training on ImageNet), '
                        'or the path to the weights file to be loaded.')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as `"imagenet"` with `include_top`'
                     ' as true, `classes` should be 1000')


    # proper input shape check

    input_shape = imagenet_utils.obtain_input_shape(input_shape, default_size=default_img_size,
            min_size=32, data_format=backend.image_data_format(),
            require_flatten=include_top,weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor
    
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    def round_filters(filters, divisor=depth_divisor):
        """ round number of filters based on depth multiplier"""
        filters *= width_coeff
        new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
        #making sure round down does not go down by 10%
        if new_filters < 0.9*filters:
            new_filters += divisor
        return int(new_filters)

    def round_repeats(repeats):
        """Round number of repeats based on depth multiplier."""
        return int(math.ceil(depth_coeff * repeats))

    
    #building stem

    x = img_input
    x = layers.Rescaling(1. / 255.)(x)
    x = layers.Normalization(axis=bn_axis)(x)

    x = layers.ZeroPadding2D(padding=imagenet_utils.correct_pad(x, 3))(x)
    x = layers.Conv2D(round_filters(32), kernel_size=3, strides=2, padding='valid',
            use_bias=False, kernel_initializer=CONV_KERNEL_INITIALIZER)(x)
    
    x = layers.BatchNormalization(axis=bn_axis)(x)
    x = layers.Activation(activations)(x)

    #building blocks
    block_args = copy.deepcopy(block_args)

    b = 0
    blocks = float(sum(round_repeats(args['repeats']) for args in block_args))

    for (i, args) in enumerate(block_args):
        assert args['repeats'] > 0

        args['filters_in'] = round_filters(args['filters_in'])
        args['filters_out'] = round_filters(args['filters_out'])

        for j in range(round_repeats(args.pop('repeats'))):
            # The first block needs to take care of stride and filter size increase.
            if j > 0:
                args['strides'] = 1
                args['filters_in'] = args['filters_out']

            x = mbconvblock(inputs=x, activations=activations,
                droprate=drop_connect_rate * b / blocks,name=str(i), **args)
            b+=1
    

    #build top

    x = layers.Conv2D(round_filters(1280),kernel_size=1, padding='same',
            use_bias=False, kernel_initializer=CONV_KERNEL_INITIALIZER)(x)
    
    x = layers.BatchNormalization(axis=bn_axis)(x)
    x = layers.Activation(activations)(x)

    if include_top:
        x = layers.GlobalAveragePooling2D()(x)
        if dropoutrate > 0:
            x = layers.Dropout(dropoutrate)(x)
        imagenet_utils.validate_activation(classifier_activation, weights)

        x = layers.Dense(classes, activation=classifier_activation, kernel_initializer=DENSE_KERNEL_INITIALIZER)(x)
    
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)
    

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = layer_utils.get_source_inputs(input_tensor)
    else:
        inputs = img_input
    
    #creating model
    model = training.Model(inputs, x)

    return model
