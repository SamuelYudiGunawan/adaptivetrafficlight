import cv2
import threading
import time
import onnxruntime as ort
import logging
from inference import InferencePipeline
import numpy as np # Import numpy for array handling

# Configure logging to show INFO messages and above
# Re-enabled INFO logging as it's sufficient for main flow and will show tracker IDs
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

# Global variables (rest remain the same)
latest_frame = None
latest_predictions = []
lock = threading.Lock()
running = True
frame_count = 0
fps = 0.0
last_fps_time = time.time()
API_KEY = "P3zgdw1MMQvDEtWuLYlq"
zona_macet = set()
kendaraan_keluar = set()
G_MASUK = 300
G_KELUAR = 500
THRESHOLD_MACET = 5
lampu_hijau = False

def my_sink(result, video_frame):
    global latest_frame, latest_predictions

    if result.get("output_image"):
        with lock:
            latest_frame = result["output_image"].numpy_image

    detection_preds = result.get("detection_predictions")
    updated_preds = []

    if detection_preds and hasattr(detection_preds, 'xyxy') and len(detection_preds.xyxy) > 0:
        boxes = detection_preds.xyxy
        confidences = detection_preds.confidence
        
        classes = getattr(detection_preds, 'class_name', detection_preds.data.get('class_name', []))
        if isinstance(classes, np.ndarray):
            classes = classes.tolist()

        tracker_ids_list_raw = getattr(detection_preds, 'tracker_id', []) # Directly access 'tracker_id' attribute
        if isinstance(tracker_ids_list_raw, np.ndarray):
            tracker_ids_list = tracker_ids_list_raw.tolist()
        else:
            tracker_ids_list = tracker_ids_list_raw # Assume it's already a list or similar
        logging.info(f"--- my_sink: Detected {len(boxes)} vehicles in this frame ---")
        
        for i in range(len(boxes)):
            label = classes[i] if i < len(classes) else "unknown"
            conf = float(confidences[i])
            x1, y1, x2, y2 = [float(coord) for coord in boxes[i]]
            
            current_tracker_id = None
            if tracker_ids_list and i < len(tracker_ids_list):
                val = tracker_ids_list[i]
                if val is not None:
                    try:
                        current_tracker_id = int(val)
                    except (ValueError, TypeError):
                        logging.warning(f"  Could not convert tracker_id '{val}' to int for index {i}. Setting to None.")
                        current_tracker_id = None
            
            logging.info(f"  Vehicle {i}: Label='{label}', Conf={conf:.2f}, Box=({x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}), TrackerID={current_tracker_id}")

            updated_preds.append({
                "class": label,
                "confidence": conf,
                "bounding_box": {
                    "x": x1,
                    "y": y1,
                    "width": x2 - x1,
                    "height": y2 - y1
                },
                "tracker_id": current_tracker_id
            })

    with lock:
        latest_predictions = updated_preds

def display_loop():
    global running, frame_count, fps, last_fps_time, lampu_hijau, zona_macet, kendaraan_keluar

    try:
        while running:
            with lock:
                frame = latest_frame.copy() if latest_frame is not None else None
                predictions = list(latest_predictions)

            if frame is not None:
                frame_height, frame_width, _ = frame.shape

                cv2.line(frame, (0, G_MASUK), (frame_width, G_MASUK), (255, 255, 0), 2)
                cv2.line(frame, (0, G_KELUAR), (frame_width, G_KELUAR), (0, 0, 255), 2)
                cv2.putText(frame, f"Masuk (Y={G_MASUK})", (10, G_MASUK - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame, f"Keluar (Y={G_KELUAR})", (10, G_KELUAR - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                current_frame_vehicles_in_jam_condition = 0

                for pred in predictions:
                    bbox = pred.get("bounding_box", {})
                    label = pred.get("class", "unknown")
                    conf = round(pred.get("confidence", 0) * 100, 1)
                    tracker_id = pred.get("tracker_id", None)

                    x1 = int(bbox.get("x", 0))
                    y1 = int(bbox.get("y", 0))
                    w = int(bbox.get("width", 0))
                    h = int(bbox.get("height", 0))
                    x2, y2 = x1 + w, y1 + h

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label_text = f"{label} ({conf}%) ID:{tracker_id if tracker_id is not None else 'N/A'}"
                    cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    if tracker_id is not None:
                        if y2 > G_MASUK and tracker_id not in kendaraan_keluar:
                            zona_macet.add(tracker_id)
                            current_frame_vehicles_in_jam_condition += 1
                            cv2.putText(frame, f"IN JAM ({y2:.0f})", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                        else:
                            if y2 <= G_MASUK:
                                cv2.putText(frame, f"NOT IN JAM (y2<M, y2={y2:.0f})", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 255), 1)
                            elif tracker_id in kendaraan_keluar:
                                cv2.putText(frame, f"ALREADY EXITED", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 100, 255), 1)

                        if y1 > G_KELUAR:
                            if tracker_id not in kendaraan_keluar:
                                kendaraan_keluar.add(tracker_id)
                                logging.info(f"Tracker ID {tracker_id} (y1={y1:.1f}) crossed G_KELUAR ({G_KELUAR}), added to kendaraan_keluar.")
                            if tracker_id in zona_macet:
                                zona_macet.discard(tracker_id)
                                logging.info(f"Tracker ID {tracker_id} discarded from zona_macet (crossed G_KELUAR).")

                macet = len(zona_macet) > THRESHOLD_MACET
                lampu_hijau = macet

                status_text = "LAMPU HIJAU - MACET" if lampu_hijau else "LALU LINTAS LANCAR"
                warna_status = (0, 255, 255) if lampu_hijau else (0, 255, 0)
                cv2.putText(frame, status_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, warna_status, 3)
                
                num_vehicles_in_jam_zone = len(zona_macet)
                cv2.putText(frame, f"Kendaraan di Zona Macet: {num_vehicles_in_jam_zone}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                logging.info(f"Current vehicles in zona_macet (SET): {sorted(list(zona_macet))}")
                logging.info(f"Tracker IDs that have exited (kendaraan_keluar SET): {sorted(list(kendaraan_keluar))}")
                logging.info(f"Is traffic jammed? {macet} (THRESHOLD_MACET={THRESHOLD_MACET})")

                frame_count += 1
                current_time = time.time()
                if current_time - last_fps_time >= 1.0:
                    fps = frame_count / (current_time - last_fps_time)
                    last_fps_time = current_time
                    frame_count = 0
                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow("ONNX Inference Stream", frame)

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                logging.info("User requested exit.")
                running = False
                break

        cv2.destroyAllWindows()
    except Exception as e:
        logging.error(f"Error in display_loop: {e}")
    finally:
        running = False

def run_pipeline(pipeline):
    try:
        pipeline.start()
        logging.info("InferencePipeline started.")
    except Exception as e:
        logging.error(f"Error in pipeline thread: {e}")
    finally:
        logging.info("InferencePipeline thread finished.")

def main():
    global running

    try:
        providers = ort.get_available_providers()
        logging.info(f"Available ONNX Runtime providers: {providers}")
        if "CUDAExecutionProvider" not in providers:
            logging.warning("CUDAExecutionProvider not found. Inference might be slower. Using CPUExecutionProvider.")

        pipeline = InferencePipeline.init_with_workflow(
            api_key=API_KEY,
            workspace_name="adaptive-traffic-light-v2",
            workflow_id="detect-and-classify-2", # Check if this workflow ID is still correct after Roboflow changes
            video_reference="videotest1edited.mp4", # Path to video, device id (int, usually 0 for built in webcams), or RTSP stream url
            max_fps=30,
            on_prediction=my_sink
        )
        logging.info("InferencePipeline initialized.")

        pipeline_thread = threading.Thread(target=run_pipeline, args=(pipeline,))
        display_thread = threading.Thread(target=display_loop)

        pipeline_thread.start()
        display_thread.start()

        display_thread.join()
        logging.info("Display thread finished. Signalling pipeline to stop.")
        running = False

        if hasattr(pipeline, "stop"):
            pipeline.stop()
            logging.info("Pipeline stop method called.")

        pipeline_thread.join()
        logging.info("Pipeline thread finished. Shutdown complete.")

    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received. Exiting...")
        running = False
        if 'pipeline' in locals() and hasattr(pipeline, "stop"):
            pipeline.stop()
    except Exception as e:
        logging.error(f"Unexpected error in main: {e}")
    finally:
        if running:
            running = False
            logging.info("Forcing application shutdown.")

if __name__ == "__main__":
    main()