import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import time

# Configuración de rutas
aves_path = '/home/lcc_rn_21/Aves/Dataset_Aves'
train_directory = os.path.join(aves_path, 'train')
test_directory = os.path.join(aves_path, 'test')
valid_directory = os.path.join(aves_path, 'valid')

# Verificar rutas
print("Verificando rutas del dataset...")
print(f"Train: {os.path.exists(train_directory)}")
print(f"Test: {os.path.exists(test_directory)}")
print(f"Valid: {os.path.exists(valid_directory)}")

# Parámetros
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 25

def create_resnet_model():
    """
    Crea modelo con ResNet50 pre-entrenado y capas personalizadas
    """
    # Cargar ResNet50 pre-entrenado (excluyendo capas fully-connected)
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Congelar las capas del modelo base
    base_model.trainable = False

    # Añadir capas personalizadas
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)

    # Obtener número de clases del dataset
    num_classes = len(os.listdir(train_directory))
    predictions = Dense(num_classes, activation='softmax')(x)

    # Modelo completo
    model = Model(inputs=base_model.input, outputs=predictions)

    return model, base_model, num_classes

def create_data_generators():
    """
    Crea generadores de datos con preprocesamiento de ResNet
    """
    # Data augmentation para training
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    # Solo preprocesamiento para validation y test
    valid_test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    # Generadores
    train_generator = train_datagen.flow_from_directory(
        train_directory,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    valid_generator = valid_test_datagen.flow_from_directory(
        valid_directory,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    test_generator = valid_test_datagen.flow_from_directory(
        test_directory,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, valid_generator, test_generator

def fine_tune_model(model, base_model, train_generator, valid_generator):
    """
    Fase de fine-tuning: descongelar capas finales
    """
    # Descongelar las últimas capas de ResNet
    base_model.trainable = True

    # Congelar primeras capas, descongelar últimas
    for layer in base_model.layers[:100]:
        layer.trainable = False
    for layer in base_model.layers[100:]:
        layer.trainable = True

    # Recompilar con learning rate más bajo
    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def main():
    start_time = time.time()
    print("Iniciando Transfer Learning con ResNet50...")

    # Crear generadores
    print("Creando generadores de datos...")
    train_generator, valid_generator, test_generator = create_data_generators()

    # Crear modelo
    print("Creando modelo ResNet50...")
    model, base_model, num_classes = create_resnet_model()

    # Compilar modelo
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(f"Modelo creado para {num_classes} clases")
    model.summary()

    # Callbacks
        # Callbacks Fase 1 (todos monitorean val_accuracy)
    callbacks = [
        EarlyStopping(patience=8, restore_best_weights=True, verbose=1, monitor='val_accuracy'),
        ReduceLROnPlateau(factor=0.2, patience=4, verbose=1, monitor='val_accuracy'),
        ModelCheckpoint('best_modelo_aves.keras', save_best_only=True, monitor='val_accuracy', verbose=1)
    ]

    print("=== FASE 1: Entrenando capas personalizadas ===")
    history1 = model.fit(
        train_generator,
        epochs=10,
        validation_data=valid_generator,
        callbacks=callbacks,
        verbose=1
    )

    print("\n=== FASE 2: Fine-tuning ===")
    model = fine_tune_model(model, base_model, train_generator, valid_generator)

    # Reinstanciar callbacks para Fase 2
    callbacks_ft = [
        EarlyStopping(patience=8, restore_best_weights=True, verbose=1, monitor='val_accuracy'),
        ReduceLROnPlateau(factor=0.2, patience=4, verbose=1, monitor='val_accuracy'),
        ModelCheckpoint('best_modelo_aves.keras', save_best_only=True, monitor='val_accuracy', verbose=1)
    ]

    history2 = model.fit(
        train_generator,
        epochs=15,
        validation_data=valid_generator,
        callbacks=callbacks_ft,
        verbose=1
    )

    # Combinar historiales
    history = {
        'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
        'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
        'loss': history1.history['loss'] + history2.history['loss'],
        'val_loss': history1.history['val_loss'] + history2.history['val_loss']
    }

    print("\n=== EVALUACIÓN FINAL ===")
    test_loss, test_accuracy = model.evaluate(test_generator, verbose=1)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")

    # Warm-up para fijar input shape
    _ = model(tf.zeros([1, 224, 224, 3], dtype=tf.float32))

    # Guardar modelo final (formato moderno)
    model.save('modelo_aves.keras')
    print("Modelo guardado como 'modelo_aves.keras'")


    import json
    with open('classes_modelo_aves.json', 'w', encoding='utf-8') as f:
        json.dump(list(train_generator.class_indices.keys()), f,
              ensure_ascii=False, indent=2)
    # Graficar resultados
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['accuracy'], label='Train Accuracy')
    plt.plot(history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('resnet_training_history.png', dpi=300, bbox_inches='tight')
    print("Gráficas guardadas como 'resnet_training_history.png'")

    # Información de clases
    class_indices = train_generator.class_indices
    classes = list(class_indices.keys())
    print(f"\nClases del dataset: {classes}")

    total_time = time.time() - start_time
    print(f"\nTiempo total de ejecución: {total_time/60:.2f} minutos")

if __name__ == "__main__":
    main()