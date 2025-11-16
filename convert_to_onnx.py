import tensorflow as tf
import tf2onnx

# Ruta de tu modelo Keras
KERAS_MODEL_PATH = "final_resnet_aves_model.keras"
ONNX_MODEL_PATH  = "modelo2/aves_resnet.onnx"

def main():
    print("Cargando modelo Keras...")
    model = tf.keras.models.load_model(KERAS_MODEL_PATH)
    model.summary()

    # Tu modelo usa imágenes 224x224x3 (según el summary que mandaste)
    input_signature = (
        tf.TensorSpec(
            (None, 224, 224, 3),  # batch size variable, 224x224x3
            tf.float32,
            name="input"          # este será el nombre de la entrada en ONNX
        ),
    )

    print("Convirtiendo a ONNX...")
    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=13,                # versión de ONNX, 13 es estable
        output_path=ONNX_MODEL_PATH
    )

    print(f"Conversión completa. ONNX guardado en: {ONNX_MODEL_PATH}")

if __name__ == "__main__":
    main()
