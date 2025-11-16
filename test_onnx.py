import numpy as np
import onnx
import onnxruntime as ort

ONNX_MODEL_PATH = "modelo2/aves_resnet.onnx"

def main():
    print("Cargando modelo ONNX...")
    model = onnx.load(ONNX_MODEL_PATH)
    onnx.checker.check_model(model)
    print("Modelo ONNX es válido ✅")

    print("\nEntradas del modelo:")
    for inp in model.graph.input:
        print(" -", inp.name)

    print("\nSalidas del modelo:")
    for out in model.graph.output:
        print(" -", out.name)

    # Crear sesión de inferencia
    session = ort.InferenceSession(ONNX_MODEL_PATH, providers=["CPUExecutionProvider"])

    input_info = session.get_inputs()[0]
    input_name = input_info.name
    input_shape = input_info.shape

    print("\nNombre real de la entrada:", input_name)
    print("Shape esperado de la entrada:", input_shape)

    # Asumimos que es [None, 224, 224, 3]; creamos un dummy
    dummy = np.random.rand(1, 224, 224, 3).astype("float32")

    outputs = session.run(None, {input_name: dummy})
    print("\nNúmero de tensores de salida:", len(outputs))
    print("Shape de la primera salida:", outputs[0].shape)

if __name__ == "__main__":
    main()
