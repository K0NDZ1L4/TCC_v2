import tensorflow as tf
from sklearn.model_selection import train_test_split






class Cnn():

    def __new__(cls, features,target):
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        # Construir o modelo CNN
        model = tf.keras.Sequential([
            tf.keras.layers.Reshape((X_train.shape[1], 1), input_shape=(X_train.shape[1],)),
            tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu'),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(64, kernel_size=2, activation='relu'),  # Reduzindo o tamanho do kernel
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # Compilar o modelo
        model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        # Treinar o modelo
        model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

        # Avaliar o modelo
        loss, accuracy = model.evaluate(X_test, y_test)
        print("Accuracy:", accuracy)





