# from ultralytics import YOLO
# model = YOLO("runs/detect/barcode_detection_v15/weights/best.pt")
# model.export(format="onnx", dynamic=True, simplify=True, opset=17)
from ultralytics import YOLO
import os

# Modelin mutlak yolu
model_path = r"C:\Users\Administrator\runs\detect\barcode_detection_v15\weights\best.pt"

# Modeli yükleme ve kontrol
if os.path.exists(model_path):
    try:
        model = YOLO(model_path)
        print("✅ Model başarıyla yüklendi!")
        print(f"Model bilgisi: {model.names}")  # Tanıyabildiği sınıfları göster
        
        # Test tahmini yapalım
        results = model.predict("test.jpg", save=True, conf=0.5)
        print("🎯 Tahmin tamamlandı! Sonuçlar 'runs/detect/predict' klasöründe.")
        
    except Exception as e:
        print(f"❌ Model yüklenirken hata: {str(e)}")
else:
    print(f"⚠️ Dosya bulunamadı: {model_path}")
    print("Mevcut dosyalar:")
    for f in os.listdir(os.path.dirname(model_path)):
        print(f" - {f}")