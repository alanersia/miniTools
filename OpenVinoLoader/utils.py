import openvino as ov
import numpy as np
import cv2
import time  # <-- Tambahkan import time

class processing:
    """
    Function:
    - Preprocessing = Create preprocessing image to input model size
    - xyxyToOriginal = Post Processing Convert xyxy to original location of images input
    - xywhToOriginal = Post Processing xywh to original location of images input
    """

    def xyxyToOriginal(boxes, r, orig_shape):
        """
        Mengonversi bounding box dari koordinat gambar yang diproses (padding) ke koordinat asli (format xyxy).

        Args:
            boxes (np.ndarray): Bounding box dalam format xyxy (N, 4)
            r (float): Rasio scaling dari preprocessing.
            orig_shape (tuple): Dimensi asli gambar (height, width).

        Returns:
            np.ndarray: Bounding box dalam koordinat asli (xyxy).
        """
        H_orig, W_orig = orig_shape
        W_resized = W_orig * r  # Lebar gambar setelah di-resize (sebelum padding)
        H_resized = H_orig * r  # Tinggi gambar setelah di-resize (sebelum padding)

        # Clip bounding box agar tidak melebihi area gambar yang di-resize
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, W_resized)  # Clip koordinat x
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, H_resized)  # Clip koordinat y

        # Scalling inverse untuk mendapatkan koordinat asli
        boxes = boxes / r
        return boxes
    
    def xywhToOriginal():
        """
        Mengonversi bounding box dari koordinat gambar yang diproses (padding) ke koordinat asli (format xywh).

        Args:
            boxes (list/np.ndarray): Bounding box dalam format xywh (N, 4)
            r (float): Rasio scaling dari preprocessing.
            orig_shape (tuple): Dimensi asli gambar (height, width).

        Returns:
            np.ndarray: Bounding box dalam koordinat asli (xywh).
        """
        # Konversi ke numpy array jika input berupa list
        boxes = np.array(boxes)  # <--- Tambahkan ini

        # Konversi xywh ke xyxy
        xyxy_boxes = np.zeros_like(boxes)
        xyxy_boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
        xyxy_boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
        xyxy_boxes[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
        xyxy_boxes[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2

        # Gunakan fungsi konversi xyxy
        xyxy_boxes = convert_xyxy_to_original(xyxy_boxes, r, orig_shape)

        # Konversi kembali ke xywh
        converted_boxes = np.zeros_like(xyxy_boxes)
        converted_boxes[:, 0] = (xyxy_boxes[:, 0] + xyxy_boxes[:, 2]) / 2  # x_center
        converted_boxes[:, 1] = (xyxy_boxes[:, 1] + xyxy_boxes[:, 3]) / 2  # y_center
        converted_boxes[:, 2] = xyxy_boxes[:, 2] - xyxy_boxes[:, 0]  # width
        converted_boxes[:, 3] = xyxy_boxes[:, 3] - xyxy_boxes[:, 1]  # height

        return converted_boxes
    
    def preprocessing(img, input_size, swap=(2, 0, 1)):
        if len(img.shape) == 3:
            padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
        else:
            padded_img = np.ones(input_size, dtype=np.uint8) * 114

        r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

        padded_img = padded_img.transpose(swap)
        padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
        return padded_img, r
    

class ovloader:
    """
    def __init__(self, model : str, device : str = "CPU"):
        self.model = model
        self.device = device
    """
    def loader(model, device, input_data):
        core = ov.Core
        loaded = core.read_model(model=model)
        compiled = core.compile_model(model=loaded, device_name=device)
        input_layer = loaded.input(0)
        N,C,H,W = input_layer.shape
        nchw = (N,C,H,W)
        result = compiled(input_data)
        return nchw, result


core = ov.Core()
classification_model_xml = "ovmodels/ppe/ppev1/yolo_nass_s.xml"

model = core.read_model(model=classification_model_xml)
compiled_model = core.compile_model(model=model, device_name="GPU")
input_layer = model.input(0)
N, C, H, W = input_layer.shape

cap = cv2.VideoCapture(0)

# Variabel untuk menghitung FPS
prev_time = 0  # <-- Inisialisasi variabel waktu


def convert_xywh_to_original(boxes, r, orig_shape):
    """
    Mengonversi bounding box dari koordinat gambar yang diproses (padding) ke koordinat asli (format xywh).

    Args:
        boxes (list/np.ndarray): Bounding box dalam format xywh (N, 4)
        r (float): Rasio scaling dari preprocessing.
        orig_shape (tuple): Dimensi asli gambar (height, width).

    Returns:
        np.ndarray: Bounding box dalam koordinat asli (xywh).
    """
    # Konversi ke numpy array jika input berupa list
    boxes = np.array(boxes)  # <--- Tambahkan ini

    # Konversi xywh ke xyxy
    xyxy_boxes = np.zeros_like(boxes)
    xyxy_boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
    xyxy_boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
    xyxy_boxes[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2
    xyxy_boxes[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2

    # Gunakan fungsi konversi xyxy
    xyxy_boxes = convert_xyxy_to_original(xyxy_boxes, r, orig_shape)

    # Konversi kembali ke xywh
    converted_boxes = np.zeros_like(xyxy_boxes)
    converted_boxes[:, 0] = (xyxy_boxes[:, 0] + xyxy_boxes[:, 2]) / 2  # x_center
    converted_boxes[:, 1] = (xyxy_boxes[:, 1] + xyxy_boxes[:, 3]) / 2  # y_center
    converted_boxes[:, 2] = xyxy_boxes[:, 2] - xyxy_boxes[:, 0]  # width
    converted_boxes[:, 3] = xyxy_boxes[:, 3] - xyxy_boxes[:, 1]  # height

    return converted_boxes

def preprocessing(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r

def draw_bounding_box(frame, bbox, color=(0, 255, 0), label=None):
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if label:
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

while True:
    success, img = cap.read()
    if not success:
        print('[INFO] Gagal membaca frame...')
        break

    # Mulai menghitung waktu processing
    start_time = time.time()  # <-- Waktu mulai processing

    orig_height, orig_width = img.shape[:2]
    orig_shape = (orig_height, orig_width)
    input_data, ratio = preprocessing(img, (H, W))
    input_data = np.expand_dims(np.transpose(input_data, (0, 1, 2)), 0).astype(np.float32)

    # Inferensi
    result = compiled_model(input_data)

    # Post-processing
    endresult = []
    for res in result[0]:
        if res[0] != -1:
            boxes = [res[1], res[2], res[3], res[4]]
            boxes = [x.item() for x in boxes]
            endresult.append(boxes)

    converted_boxes = convert_xywh_to_original(endresult, ratio, orig_shape)
    results = []
    for box in converted_boxes:
        x1, y1, x2, y2 = box
        results.append([int(x1), int(y1), int(x2), int(y2)])

    # Gambar bounding box
    if len(results):
        for res in results:
            draw_bounding_box(img, res, (0, 255, 0))

    # Hitung FPS
    current_time = time.time()
    fps = 1 / (current_time - start_time)  # <-- Hitung FPS
    fps_text = f"FPS: {fps:.2f}"  # <-- Format teks FPS

    # Tampilkan FPS di frame
    cv2.putText(img, fps_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # <-- Tambahkan teks FPS

    cv2.imshow('YOLO-NAS, OPENVINO', img)

    # Keluar jika tombol 'q' ditekan
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()