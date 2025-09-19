from ultralytics import YOLO
import os
from multiprocessing import freeze_support

def main():
    # Dosya yolu ayarları
    DATA_YAML_PATH = r"C:\Users\Administrator\Desktop\BarkodTanima\barcode-detection\data.yaml"
    
    if not os.path.exists(DATA_YAML_PATH):
        raise FileNotFoundError(f"data.yaml dosyası bulunamadı: {DATA_YAML_PATH}")

    # Modeli yükle ve eğit
    model = YOLO("yolov8n.pt")
    results = model.train(
        data=DATA_YAML_PATH,
        epochs=100,
        imgsz=640,
        batch=16,
        device="0",
        name="barcode_detection_v1",
        workers=4  # Multiprocessing için worker sayısı
    )
    print("Eğitim başarıyla tamamlandı!")

if __name__ == '__main__':
    freeze_support()  # Windows'ta çoklu işlem için gerekli
    main()