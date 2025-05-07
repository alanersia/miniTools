import numpy as np
import openvino as ov

# 1. Buat Core object
core = ov.Core()
model_path = "OpenVinoLoader/model/yolox_tiny.xml"

# 2. Baca model IR (xml + bin otomatis dipadankan)
model = core.read_model(model=model_path)

# 3. Kompilasi model ke device
compiled_model = core.compile_model(model=model, device_name="GPU")
# 4. Ambil nama input/output
input_port = compiled_model.input(0)
output_port = compiled_model.output(0)

# 5. Jalankan inference
image = np.random.randn(1, 3, 416, 416).astype(np.float32)
result = compiled_model([image])[output_port]
print(result)