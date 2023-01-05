import tensorflow as tf


class char_language_model:

    def __init__(self, vocab_size, number_of_lang, char_vector_dim, lang_vector_dim, hidden_size):
        ##### ##### ##### ##### MODEL #####  ##### ##### ##### #####

        self.lang_embedding_layer = tf.keras.layers.Embedding(number_of_lang, lang_vector_dim, mask_zero=True, name="language_vectors")
        self.char_embedding_layer = tf.keras.layers.Embedding(vocab_size, char_vector_dim, mask_zero=True, name="char_vectors")

        self.lstm = tf.keras.layers.LSTM(units=hidden_size, input_shape=(None, char_vector_dim + lang_vector_dim), return_sequences=True)
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.dense = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(vocab_size, activation='softmax'))

        ## INPUTS

        input_char_vector = tf.keras.Input(shape=(None,), name="char_input_layer")
        input_lang_vector = tf.keras.Input(shape=(None,), name="lang_input_layer")

        # forward
        char_embeddings = self.char_embedding_layer(input_char_vector)
        language_embedddings = self.lang_embedding_layer(input_lang_vector)

        merged = tf.keras.layers.concatenate([char_embeddings, language_embedddings])
        lstm_out = self.lstm(merged)
        dropped = self.dropout(lstm_out)
        out = self.dense(dropped)

        model = tf.keras.models.Model(inputs=[input_char_vector, input_lang_vector], outputs=out)
        model.summary()
        model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])

        self.model = model
