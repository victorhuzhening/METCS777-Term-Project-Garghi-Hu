import warnings

warnings.filterwarnings('ignore', category=UserWarning)
import tensorflow as tf
import numpy as np
import string
import re
import tensorflow_addons as tfa

FRAME_LEN = 250
vocab_size = 1085
sequence_length = 20
pad_token_idx = 0


@tf.function
def normalize(keypoints):
    keypoints = tf.reshape(keypoints, [-1, keypoints.shape[1] * keypoints.shape[2]])
    mean, variance = tf.nn.moments(keypoints, [-1], keepdims=True)
    normalized_keypoints = tf.nn.batch_normalization(keypoints, mean, variance, offset=None, scale=None,
                                                     variance_epsilon=1e-6)
    return normalized_keypoints


@tf.function
def resize_pad(x):
    num_frames = tf.shape(x)[0]
    if num_frames < FRAME_LEN:
        x = tf.pad(x, ([[0, FRAME_LEN - num_frames], [0, 0], [0, 0]]), constant_values=0.0)
    else:
        x = tf.image.resize(x, [FRAME_LEN, tf.shape(x)[1]], method=tf.image.ResizeMethod.AREA)
    return x


def standardize_caption(s):
    s = tf.strings.upper(s)
    s = tf.strings.regex_replace(s, f'[{re.escape(string.punctuation)}]', '')
    return s


# Carregar os pesos do tokenizador de um arquivo numpy
loaded_weights = np.load('saved_model/tokenizer_weights.npy', allow_pickle=True)

# Criar um novo tokenizador com a mesma configuração
tokenizer = tf.keras.layers.TextVectorization(
    max_tokens=vocab_size,
    standardize=standardize_caption,
    output_mode="int",
    output_sequence_length=sequence_length,
    ragged=False
)

# Definir os pesos no novo tokenizador
tokenizer.set_weights(loaded_weights)


class ECA(tf.keras.layers.Layer):
    def __init__(self, kernel_size=5, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.kernel_size = kernel_size
        self.conv = tf.keras.layers.Conv1D(1, kernel_size=kernel_size, strides=1, padding="same", use_bias=False)

    def call(self, inputs, mask=None):
        """
        Realiza uma operação ECA (Enhanced Channel Attention) em tensores de entrada.

        Args:
            inputs (tf.Tensor): Tensor de entrada.
            mask (tf.Tensor, opcional): Tensor de máscara para suportar sequências com comprimentos diferentes.

        Returns:
            tf.Tensor: Tensor após a operação ECA.
        """
        nn = tf.keras.layers.GlobalAveragePooling1D()(inputs, mask=mask)
        nn = tf.expand_dims(nn, -1)
        nn = self.conv(nn)
        nn = tf.squeeze(nn, -1)
        nn = tf.nn.sigmoid(nn)
        nn = nn[:, None, :]
        return inputs * nn


class MaskingDWConv1D(tf.keras.layers.Layer):
    '''
    masked DW1Dconv with strides>1, padding=same.
    NOTE: padded(masked) frames should always be at the beginning or end of the input sequence.
    '''

    def __init__(self, kernel_size, strides=1,
                 dilation_rate=1,
                 padding='same',
                 use_bias=False,
                 kernel_initializer='glorot_uniform', **kwargs):
        super().__init__(**kwargs)
        assert padding == 'same' or padding == 'causal'
        self.strides = strides
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.use_bias = use_bias
        self.padding = padding
        self.conv = tf.keras.layers.DepthwiseConv1D(
            kernel_size,
            strides=strides,
            dilation_rate=dilation_rate,
            padding=padding,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        if mask is not None:
            if self.strides > 1:
                mask = mask[:, ::self.strides]
        return mask

    def call(self, inputs, mask=None):
        x = inputs
        if mask is not None:
            x = tf.where(mask[..., None], x, tf.constant(0., dtype=x.dtype))
        x = self.conv(x)
        return x


def Conv1DBlock(channel_size,
                kernel_size,
                dilation_rate=1,
                drop_rate=0.0,
                expand_ratio=2,
                se_ratio=0.25,
                activation='tanh',
                name=None):
    '''
    Efetua uma operação de bloco conv1d eficiente.

    Args:
        channel_size (int): Número de canais de saída.
        kernel_size (int): Tamanho do kernel da convolução.
        dilation_rate (int, opcional): Taxa de dilatação para convolução causal. Padrão é 1.
        drop_rate (float, opcional): Taxa de dropout. Padrão é 0.0.
        expand_ratio (int, opcional): Fator de expansão do canal. Padrão é 2.
        se_ratio (float, opcional): Taxa de excitação espacial (SE). Padrão é 0.25.
        activation (str, opcional): Função de ativação. Padrão é 'swish'.
        name (str, opcional): Nome da camada. Padrão é None.

    Returns:
        Callable: Função que aplica o bloco conv1d eficiente.
    '''
    if name is None:
        name = str(tf.keras.backend.get_uid("mbblock"))

    # Fase de expansão
    def apply(inputs):
        channels_in = tf.keras.backend.int_shape(inputs)[-1]
        channels_expand = channels_in * expand_ratio

        skip = inputs

        x = tf.keras.layers.Dense(
            channels_expand,
            use_bias=True,
            activation=activation,
            name=name + '_expand_conv')(inputs)

        # Convolução Depthwise
        x = MaskingDWConv1D(kernel_size,
                            dilation_rate=dilation_rate,
                            use_bias=False,
                            name=name + '_dwconv')(x)

        x = tf.keras.layers.BatchNormalization(momentum=0.95, name=name + '_bn')(x)

        x = ECA()(x)

        x = tf.keras.layers.Dense(
            channel_size,
            use_bias=True,
            name=name + '_project_conv')(x)

        if drop_rate > 0:
            x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1), name=name + '_drop')(x)

        if (channels_in == channel_size):
            x = tf.keras.layers.add([x, skip], name=name + '_add')
        return x

    return apply


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, dim=256, num_heads=4, dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.scale = self.dim ** -0.5
        self.num_heads = num_heads
        self.qkv = tf.keras.layers.Dense(3 * dim, use_bias=False)
        self.drop1 = tf.keras.layers.Dropout(dropout)
        self.proj = tf.keras.layers.Dense(dim, use_bias=False)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        qkv = self.qkv(inputs)
        qkv = tf.keras.layers.Permute((2, 1, 3))(
            tf.keras.layers.Reshape((-1, self.num_heads, self.dim * 3 // self.num_heads))(qkv))
        q, k, v = tf.split(qkv, [self.dim // self.num_heads] * 3, axis=-1)

        # Calcula a atenção.
        attn = tf.matmul(q, k, transpose_b=True) * self.scale

        if mask is not None:
            mask = mask[:, None, None, :]

        attn = tf.keras.layers.Softmax(axis=-1)(attn, mask=mask)
        attn = self.drop1(attn)

        x = attn @ v
        x = tf.keras.layers.Reshape((-1, self.dim))(
            tf.keras.layers.Permute((2, 1, 3))(x))
        x = self.proj(x)
        return x


def TransformerBlock(dim=256, num_heads=4, expand=4, attn_dropout=0.2, drop_rate=0.2, activation='swish'):
    """
    Bloco de Transformer personalizado.

    Args:
        dim (int): Dimensão do espaço de características.
        num_heads (int): Número de cabeças de atenção multi-head.
        expand (int): Fator de expansão para a camada densa interna.
        attn_dropout (float): Taxa de dropout para a camada de atenção multi-head.
        drop_rate (float): Taxa de dropout para as camadas de dropout.
        activation (str): Função de ativação para as camadas densas internas.

    Returns:
        Callable: Função que aplica o bloco de Transformer a um tensor de entrada.
    """

    def apply(inputs):
        reshaped_inputs = tf.keras.layers.Dense(dim, use_bias=False)(inputs)
        x = reshaped_inputs
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        x = MultiHeadSelfAttention(dim=dim, num_heads=num_heads, dropout=attn_dropout)(x)
        x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1))(x)
        x = tf.keras.layers.Add()([reshaped_inputs, x])
        attn_out = x

        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        x = tf.keras.layers.Dense(dim * expand, use_bias=False, activation=activation)(x)
        x = tf.keras.layers.Dense(dim, use_bias=False)(x)
        x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1))(x)
        x = tf.keras.layers.Add()([attn_out, x])
        return x

    return apply


def positional_encoding(maxlen, num_hid):
    """
    Gera a codificação posicional para sequências de entrada.

    Args:
        maxlen (int): Comprimento máximo da sequência.
        num_hid (int): Número de dimensões ocultas para a codificação posicional.

    Returns:
        tf.Tensor: Codificação posicional para a sequência de entrada.
    """
    depth = num_hid / 2
    positions = tf.range(maxlen, dtype=tf.float32)[..., tf.newaxis]
    depths = tf.range(depth, dtype=tf.float32)[np.newaxis, :] / depth
    angle_rates = tf.math.divide(1, tf.math.pow(tf.cast(10000, tf.float32), depths))
    angle_rads = tf.linalg.matmul(positions, angle_rates)

    # Calcula as funções trigonométricas para a codificação posicional.
    sin_vals = tf.math.sin(angle_rads)
    cos_vals = tf.math.cos(angle_rads)

    # Concatena as funções seno e cosseno para formar a codificação posicional.
    pos_encoding = tf.concat([sin_vals, cos_vals], axis=-1)
    return pos_encoding


class CTCLoss(tf.keras.losses.Loss):
    def __init__(self, blank_index=0, input_padding_value=0., target_padding_value=0, **kwargs):
        super().__init__(**kwargs)
        self.blank_index = blank_index
        self.input_padding_value = input_padding_value
        self.target_padding_value = target_padding_value

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.float32)
        batch_len = tf.cast(tf.shape(y_true)[0], dtype=tf.int32)
        label_length = y_true != tf.cast(self.target_padding_value, tf.int32)
        label_length = tf.reduce_sum(tf.cast(label_length, tf.int32), axis=1, keepdims=False)  # (B,)
        mask = getattr(y_pred, '_keras_mask', None)
        if mask is not None:
            input_length = tf.reduce_sum(tf.cast(mask, tf.int32), axis=-1)
        else:
            input_length = tf.cast(tf.shape(y_pred)[1], dtype=tf.int32)
            input_length = input_length * tf.ones(shape=(batch_len,), dtype=tf.int32)

        loss = tf.nn.ctc_loss(y_true, y_pred, label_length=label_length, logit_length=input_length, blank_index=0,
                              logits_time_major=False)

        loss = tf.reduce_mean(loss)

        return loss


def beam_search(pred, tokenizer, beam_width=5):
    decoded_batch = []
    vocab = tokenizer.get_vocabulary()

    for logit in pred:
        if len(logit.shape) == 2:
            logit = tf.expand_dims(logit, 0)

        logit = tf.transpose(logit, perm=[1, 0, 2])
        decoded, log_prob = tf.nn.ctc_beam_search_decoder(
            logit,
            sequence_length=tf.fill([tf.shape(logit)[1]], tf.shape(logit)[0]),
            beam_width=beam_width
        )

        dense_decoded = tf.sparse.to_dense(decoded[0], default_value=pad_token_idx)
        dense_decoded = dense_decoded.numpy()
        dense_decoded = [seq[seq != pad_token_idx] for seq in dense_decoded]

        decoded_text = [' '.join([vocab[idx] for idx in seq]) for seq in dense_decoded]

        # Se todas as sequências estiverem vazias, tenta encontrar as três palavras mais prováveis
        if not any(decoded_text):
            generated_words = []
            for time_step in logit.numpy():
                top_indices = np.argsort(time_step[0])[-3:][::-1]  # Busca as três palavras mais prováveis
                for idx in top_indices:
                    if idx != pad_token_idx and len(generated_words) < 3:
                        word = vocab[idx]
                        if word:  # Verifica se a palavra encontrada não é vazia
                            generated_words.append(word)
                    if len(generated_words) == 3:
                        break
                if len(generated_words) == 3:
                    break
            decoded_text = [' '.join(generated_words)] if generated_words else ["Texto padrão"]

        decoded_batch.append(decoded_text)

    return decoded_batch


def remove_consecutive_duplicates(text):
    """
    Remove palavras duplicadas consecutivas de uma string.

    :param text: String original.
    :return: String com duplicatas consecutivas removidas.
    """
    words = text.split()
    filtered_words = []

    previous_word = None
    for word in words:
        if word != previous_word:
            filtered_words.append(word)
            previous_word = word

    return ' '.join(filtered_words)


def correcoes(text, file_path):
    def load_words_from_file(file_path):
        with open(file_path, 'r') as file:
            return [line.strip() for line in file]

    words_with_hyphen = load_words_from_file(file_path)

    for word in words_with_hyphen:
        word_without_hyphen = word.replace('-', '').upper()
        text = text.replace(word_without_hyphen, word)

    return text


def load_custom_transformer_model(model_path):
    custom_objects = {
        'ECA': ECA,
        'CausalDWConv1D': MaskingDWConv1D,
        'Conv1DBlock': Conv1DBlock,
        'MultiHeadSelfAttention': MultiHeadSelfAttention,
        'TransformerBlock': TransformerBlock,
        'positional_encoding': positional_encoding,
        'CTCLoss': CTCLoss,
        'AdamW': tfa.optimizers.AdamW
    }

    # Carregar o modelo salvo
    loaded_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)

    return loaded_model