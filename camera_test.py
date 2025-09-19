from ultralytics import YOLO
import cv2
import time

# 1. Model Yükleme
model_path = r"C:\Users\Administrator\runs\detect\barcode_detection_v15\weights\best.pt"
model = YOLO(model_path)
print("✅ Model yüklendi! Tanıyabildiği sınıflar:", model.names)

# 2. Kamera Ayarları
cap = cv2.VideoCapture(0)  # 0 = Varsayılan kamera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 3. Performans Metrikleri
fps_counter = 0
start_time = time.time()

while True:
    # Kameradan görüntü al
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Kamera görüntüsü alınamadı!")
        break
    
    # 4. Tahmin Yap
    results = model(frame, stream=True, conf=0.7, verbose=False)  # verbose=False daha temiz çıktı
    
    # 5. Görselleştirme
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Kutu çiz
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Etiket yaz
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = f"{model.names[cls_id]} %{conf*100:.1f}"
            cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
    
    # 6. FPS Hesapla
    fps_counter += 1
    if fps_counter % 10 == 0:
        fps = fps_counter / (time.time() - start_time)
        cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    
    # 7. Görüntüyü göster
    cv2.imshow("Barkod Tanıma - Çıkmak için 'q' basın", frame)
    
    # 8. Çıkış kontrolü
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Temizlik
cap.release()
cv2.destroyAllWindows()
print("❌ Kameradan çıkıldı")