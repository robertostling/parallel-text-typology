import tensorflow as tf


class predict_we_language_model:

    def __init__(self, vocab_size, number_of_lang, char_vector_dim, lang_vector_dim, hidden_size):
        ##### ##### ##### ##### MODEL #####  ##### ##### ##### #####

        self.lang_embedding_layer = tf.keras.layers.Embedding(number_of_lang, lang_vector_dim, mask_zero=True, name="language_vectors")
        self.char_embedding_layer = tf.keras.layers.Embedding(vocab_size, char_vector_dim, mask_zero=True, name="char_vectors")

        self.lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=hidden_size, input_shape=(None, char_vector_dim + lang_vector_dim), return_sequences=True))
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.dense = tf.keras.layers.Dense(300, activation='softmax')

        ## INPUTS

        input_char_vector = tf.keras.Input(shape=(None,), name="char_input_layer")
        input_lang_vector = tf.keras.Input(shape=(None,), name="lang_input_layer")

        # forward
        char_embeddings = self.char_embedding_layer(input_char_vector)
        language_embedddings = self.lang_embedding_layer(input_lang_vector)

        merged = tf.keras.layers.concatenate([char_embeddings, language_embedddings])
        lstm_out = self.lstm(merged)

        attention = tf.keras.layers.Dense(1, activation='tanh')(lstm_out)
        attention = tf.keras.layers.Flatten()(attention)
        attention = tf.keras.layers.Activation('softmax')(attention)
        attention = tf.keras.layers.RepeatVector(hidden_size*2)(attention)
        attention = tf.keras.layers.Permute([2, 1])(attention)

        attention_out = tf.keras.layers.Multiply()([lstm_out, attention])
        sent_representation = tf.keras.layers.Lambda(lambda xin: tf.keras.backend.sum(xin, axis=-2), output_shape=(300,))(attention_out)

        dropped = self.dropout(sent_representation)
        out = self.dense(dropped)
        print(out.shape)

        model = tf.keras.models.Model(inputs=[input_char_vector, input_lang_vector], outputs=out)
        model.summary()
        loss = lambda x, y: 1 - tf.keras.losses.cosine_similarity(x, y, axis=1)
        model.compile(loss=loss, optimizer='adam', metrics=["accuracy"])

        self.model = model


if __name__ == "__main__":
    x = predict_we_language_model(32,1480,100,100,128)
    total_parameters = 0
    for variable in x.model.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        print(shape)
        print(len(shape))
        variable_parameters = 1
        for dim in shape:
            print(dim)
            variable_parameters *= dim.value
        print(variable_parameters)
        total_parameters += variable_parameters
    print(total_parameters)
    print()