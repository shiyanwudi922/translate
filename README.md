问题描述：tensorflow旧版本tutorial中的seq2seq部分的机器翻译模型在1.2版本以后会出现bug。而在更新版本的tutorial中，直接放弃了旧版本的代码（主要是因为旧版本代码使用static_rnn+buckets技术，执行效率比较低；新版本代码使用了新的dynamic_rnn接口，提高了效率）（应该是Google的人也看不上原来的代码了，所以懒得再去改），提供了新的nmt代码，但是这样会导致已经在业务中使用了旧版本代码的用户的更新代价比较高。

分析：主要还是由于embedding_attention_seq2seq这个接口在调用时对encoder和decoder共享参数，需要进行deepcopy，而源代码中这一部分实现有误造成的。

目的：这里对旧版本代码做了一些修改，使其仍然可以运行，算是一种折衷的办法
主要参考：https://github.com/tensorflow/tensorflow/issues/8191#issuecomment-311003867 这里提出的三种方法

一、在调用embedding_attention_seq2seq之前显式copy一次，参考：

Miopas commented on Jun 23
I also met the error caused by copy.deepcopy(cell) in embedding_attention_seq2seq() when running self_test() in the translate model in tutorial.
I tried to change the codes in seq2seq_f() in Seq2SeqModel as follows:

    def seq2seq_f(encoder_inputs, decoder_inputs, do_decode=False):
        tmp_cell = copy.deepcopy(cell) #new
        return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
            encoder_inputs,
            decoder_inputs,
            tmp_cell, #new
            num_encoder_symbols=source_vocab_size,
            num_decoder_symbols=target_vocab_size,
            embedding_size=size,
            output_projection=output_projection,
            feed_previous=do_decode,
            dtype=dtype)
Then there is no error now.
BUT as a rookie I don't know whether the codes here work as before and it seems the changes make the model run slower.

二、修改源码，在embedding_attention_seq2seq中分别构造encoder cell和decoder cell，而在调用的时候使用相同的cell参数， 参考：

fabiofumarola commented on Jun 26
Hi guys, I don't know if you're still interested on it, but I found that the problem is related to the operation of copying the cell passed as params to the embedding_attention_seq2seq function. This is because the same cell definition is used both for encoder and decoder. I think the tutorial is deprecated since it uses a seq2seq model with bucketing in contrast to a dynamic seq2seq. But, I'm pasting a modified function that works. The function is updated in the file tensorflow/contrib/legacy_seq2seq/python/ops/seq2seq.py.

thanks,
Fabio

def embedding_attention_seq2seq(encoder_inputs,
                                decoder_inputs,
                                enc_cell,
                                dec_cell,
                                num_encoder_symbols,
                                num_decoder_symbols,
                                embedding_size,
                                num_heads=1,
                                output_projection=None,
                                feed_previous=False,
                                dtype=None,
                                scope=None,
                                initial_state_attention=False):
  """Embedding sequence-to-sequence model with attention.

  This model first embeds encoder_inputs by a newly created embedding (of shape
  [num_encoder_symbols x input_size]). Then it runs an RNN to encode
  embedded encoder_inputs into a state vector. It keeps the outputs of this
  RNN at every step to use for attention later. Next, it embeds decoder_inputs
  by another newly created embedding (of shape [num_decoder_symbols x
  input_size]). Then it runs attention decoder, initialized with the last
  encoder state, on embedded decoder_inputs and attending to encoder outputs.

  Warning: when output_projection is None, the size of the attention vectors
  and variables will be made proportional to num_decoder_symbols, can be large.

  Args:
    encoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
    decoder_inputs: A list of 1D int32 Tensors of shape [batch_size].
    cell: tf.nn.rnn_cell.RNNCell defining the cell function and size.
    num_encoder_symbols: Integer; number of symbols on the encoder side.
    num_decoder_symbols: Integer; number of symbols on the decoder side.
    embedding_size: Integer, the length of the embedding vector for each symbol.
    num_heads: Number of attention heads that read from attention_states.
    output_projection: None or a pair (W, B) of output projection weights and
      biases; W has shape [output_size x num_decoder_symbols] and B has
      shape [num_decoder_symbols]; if provided and feed_previous=True, each
      fed previous output will first be multiplied by W and added B.
    feed_previous: Boolean or scalar Boolean Tensor; if True, only the first
      of decoder_inputs will be used (the "GO" symbol), and all other decoder
      inputs will be taken from previous outputs (as in embedding_rnn_decoder).
      If False, decoder_inputs are used as given (the standard decoder case).
    dtype: The dtype of the initial RNN state (default: tf.float32).
    scope: VariableScope for the created subgraph; defaults to
      "embedding_attention_seq2seq".
    initial_state_attention: If False (default), initial attentions are zero.
      If True, initialize the attentions from the initial state and attention
      states.

  Returns:
    A tuple of the form (outputs, state), where:
      outputs: A list of the same length as decoder_inputs of 2D Tensors with
        shape [batch_size x num_decoder_symbols] containing the generated
        outputs.
      state: The state of each decoder cell at the final time-step.
        It is a 2D Tensor of shape [batch_size x cell.state_size].
  """
  with variable_scope.variable_scope(
      scope or "embedding_attention_seq2seq", dtype=dtype) as scope:
    dtype = scope.dtype
    # Encoder.

    encoder_cell = enc_cell

    encoder_cell = core_rnn_cell.EmbeddingWrapper(
        encoder_cell,
        embedding_classes=num_encoder_symbols,
        embedding_size=embedding_size)
    encoder_outputs, encoder_state = rnn.static_rnn(
        encoder_cell, encoder_inputs, dtype=dtype)

    # First calculate a concatenation of encoder outputs to put attention on.
    top_states = [
        array_ops.reshape(e, [-1, 1, encoder_cell.output_size]) for e in encoder_outputs
    ]
    attention_states = array_ops.concat(top_states, 1)

    # Decoder.
    output_size = None
    if output_projection is None:
      dec_cell = core_rnn_cell.OutputProjectionWrapper(dec_cell, num_decoder_symbols)
      output_size = num_decoder_symbols

    if isinstance(feed_previous, bool):
      return embedding_attention_decoder(
          decoder_inputs,
          encoder_state,
          attention_states,
          dec_cell,
          num_decoder_symbols,
          embedding_size,
          num_heads=num_heads,
          output_size=output_size,
          output_projection=output_projection,
          feed_previous=feed_previous,
          initial_state_attention=initial_state_attention)

    # If feed_previous is a Tensor, we construct 2 graphs and use cond.
    def decoder(feed_previous_bool):
      reuse = None if feed_previous_bool else True
      with variable_scope.variable_scope(
          variable_scope.get_variable_scope(), reuse=reuse):
        outputs, state = embedding_attention_decoder(
            decoder_inputs,
            encoder_state,
            attention_states,
            dec_cell,
            num_decoder_symbols,
            embedding_size,
            num_heads=num_heads,
            output_size=output_size,
            output_projection=output_projection,
            feed_previous=feed_previous_bool,
            update_embedding_for_previous=False,
            initial_state_attention=initial_state_attention)
        state_list = [state]
        if nest.is_sequence(state):
          state_list = nest.flatten(state)
        return outputs + state_list

    outputs_and_state = control_flow_ops.cond(feed_previous,
                                              lambda: decoder(True),
                                              lambda: decoder(False))
    outputs_len = len(decoder_inputs)  # Outputs length same as decoder inputs.
    state_list = outputs_and_state[outputs_len:]
    state = state_list[0]
    if nest.is_sequence(encoder_state):
      state = nest.pack_sequence_as(
          structure=encoder_state, flat_sequence=state_list)
    return outputs_and_state[:outputs_len], state

三、把网络定义的语句写在embedding_attention_seq2seq调用之前， 参考：

huxuanlai commented on Jul 31
Move the code on cell definition into seq2seq_f:

def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
      def single_cell():
        return tf.contrib.rnn.GRUCell(size)
      if use_lstm:
        def single_cell():
          return tf.contrib.rnn.BasicLSTMCell(size)
      cell = single_cell()
      if num_layers > 1:
        cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in range(num_layers)])
      return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
	  ...
	  )
Then "python translate.py --data_dir data/ --train_dir checkpoint/ --size=256 --num_layers=2 --steps_per_checkpoint=50" can work.
