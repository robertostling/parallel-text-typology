import tensorflow as tf


class word_lm:

    def __init__(self, number_of_lang, word_vector_dim, lang_vector_dim, hidden_size):
        self.lang_embedding_layer = tf.keras.layers.Embedding(number_of_lang, lang_vector_dim, name="language_vectors", mask_zero=True)
        self.lstm = tf.keras.layers.LSTM(units=hidden_size, input_shape=(None, word_vector_dim + lang_vector_dim), return_sequences=True)
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(word_vector_dim, activation='linear'), activity_regularizer=tf.keras.regularizers.l2(0.01))

        # Inputs
        input_word_vector = tf.keras.Input(shape=(None, word_vector_dim), name="word_input_layer")  # word embeddings
        input_lang_vector = tf.keras.Input(shape=(None,), name="lang_input_layer")  # language ids

        # forward
        language_embedddings = self.lang_embedding_layer(input_lang_vector)
        merged = tf.keras.layers.concatenate([input_word_vector, language_embedddings])

        lstm_out = self.lstm(merged)
        dropped = self.dropout(lstm_out)
        out = self.dense(dropped)

        model = tf.keras.models.Model(inputs=[input_word_vector, input_lang_vector], outputs=out)
        model.summary()
        loss = lambda x, y: 1 - tf.keras.losses.cosine_similarity(x, y, axis=2)
        model.compile(loss=loss, optimizer="adam")
        self.model = model
