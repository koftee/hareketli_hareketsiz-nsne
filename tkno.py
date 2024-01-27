from collections import defaultdict

import cv2
import numpy as np

from ultralytics import YOLO

# yolo modelinin yüklenemsi
model = YOLO('yolov8n.pt')

# video okunacaksa okunacak videonun adresi için oluşturulan değişken
video_path = "C:\\Users\\muham\\PycharmProjects\\opencv_fotolar\\body.mp4"
cap = cv2.VideoCapture(0)

# izleme geçmişi ve belirlenen insan idleri için gerekli liste ve set
track_history = defaultdict(lambda: [])
detected_ids = set()



# her bir karenin while döngüsü ile döndürülmesi
while True:
    # frameleri okuma
    success, frame = cap.read()

    if success:
        # Durum başarılıysa yolo modelinde bulunan track fonksiyonun çalıştırılması
        results = model.track(frame, persist=True,conf=0.4,classes=0)
        human_count = 0
        uhuman_count = 0

        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # çevreleyen dikdörtgenlerin çizilmesi
        annotated_frame = results[0].plot()

        # idleri güncelleme
        detected_ids.update(track_ids)

        # arkasında bulunan çizgilerin çizimi için döngü
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box  #x,y koordinatları ve genişlik yükseklik değerlerinin eşitlenmesi
            track = track_history[track_id]     # izlenen değeri eşitleme
            track.append((int(x + w / 2), int(y + h / 2)))  # x, y için merkez nokta bulma
            if len(track) > 30:  # retain 30 tracks for 30 frames
                track.pop(0)

            # çizme kısmı
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            #verilen frame üzerine ve verilen noktalar arasına çizgiyi çizme
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=10)
            # hareketlilik tespiti için noktalar arası uzaklık bulma
            line_length = cv2.arcLength(points, closed=False)

            # 0dan büyükse hareketli kısmı arttırma değilse hareketsiz kısmı arttırma
            if line_length > 0:
                human_count+=1
            else:
                uhuman_count+=1


        new_detected_ids = detected_ids.difference(set(track_history.keys()))
        detected_ids.difference_update(new_detected_ids)


        # Sonuçları yazdırma

        cv2.putText(annotated_frame, f"moving: {human_count}", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(annotated_frame, f"unmoving: {uhuman_count}", (350, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0),
                    2)
        # işlenen frame i gösterme
        cv2.imshow("YOLOv8 Tracking", annotated_frame)

        # qya basınca durdurma
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        #eğer videodan okuyorsak video bitince döngüyü bitirme
        break



# kamerayı release etme ve tüm pencereleri yok etme
cap.release()
cv2.destroyAllWindows()
