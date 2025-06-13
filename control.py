import cv2
import numpy as np

def main():
    url = "http://192.168.137.215:4747/video"
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo conectar al stream: {url}")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

        res = controlarBrazo(frame)
        if res is not None:
            ndx, ndz = res
            print(f"Normalizado → dx: {ndx:.2f}, dz: {ndz:.2f}")

        cv2.imshow("ArUco Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_det = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
def controlarBrazo(frame):

    H, W = frame.shape[:2]
    centro_img = (W//2, H//2)

    detected = False
    # ...existing code...
    corners, ids, rejected = aruco_det.detectMarkers(frame)
    cv2.aruco.drawDetectedMarkers(frame, rejected, borderColor=(100,100,100))

    if ids is None or len(ids) == 0:
        cv2.putText(frame, "No se detectó ArUco", (10, H-30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        return None

    # Si hay detección, procesamos el primer marcador
    cv2.aruco.drawDetectedMarkers(frame, corners, ids, (0,0,255))

    # Si hay detección, calculamos dx, dz y sus versiones normalizadas
    pts = corners[0].reshape(4,2)
    cx, cy = pts.mean(axis=0).astype(int)

    dx = cx - centro_img[0]
    dz = cy - centro_img[1]

    # Normalización:
    #  - Normalizamos dx respecto a la mitad del ancho (W/2)
    #  - Normalizamos dz respecto a la mitad de la altura (H/2)
    dx_norm = dx / (W/2)   # rango aprox. [-1, 1]
    dz_norm = dz / (H/2)   # rango aprox. [-1, 1]

    # Podemos limitarlo a [-1,1] por si acaso
    dx_norm = max(-1.0, min(1.0, dx_norm))
    dz_norm = max(-1.0, min(1.0, dz_norm))

    # Dibujado para debug
    cv2.circle(frame, (cx, cy), 5, (255,0,0), -1)
    cv2.arrowedLine(frame, centro_img, (cx,cy), (255,0,0), 2, tipLength=0.2)
    cv2.putText(frame,
                f"dx={dx:+d}, dz={dz:+d}  |  ndx={dx_norm:.2f}, ndz={dz_norm:.2f}",
                (10, H-30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)
    cv2.circle(frame, centro_img, 5, (0,255,0), -1)
    cv2.putText(frame, "Centro imagen",
                (centro_img[0]+10, centro_img[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return dx_norm, dz_norm


if __name__ == "__main__":
    main()