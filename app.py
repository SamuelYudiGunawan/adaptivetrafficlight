import cv2
import threading
import time
import onnxruntime as ort
import logging
import json
from inference import InferencePipeline

# Setup logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')

# Global variables
latest_frame = None
latest_predictions = []
lock = threading.Lock()
running = True
frame_count = 0
fps = 0.0
last_fps_time = time.time()

# Your API key (temporary hardcoded)
API_KEY = "P3zgdw1MMQvDEtWuLYlq"

def my_sink(result, video_frame):
    global latest_frame, latest_predictions

    # Always update frame for smoother display
    if result.get("output_image"):
        with lock:
            latest_frame = result["output_image"].numpy_image

    # Process detection predictions
    detection_preds = result.get("detection_predictions")
    updated_preds = []

    if detection_preds and hasattr(detection_preds, 'xyxy') and len(detection_preds.xyxy) > 0:
        print("\n🔍 Detected Objects:")

        boxes = detection_preds.xyxy
        confidences = detection_preds.confidence
        classes = detection_preds.data.get('class_name', [])

        for i in range(len(boxes)):
            label = classes[i] if i < len(classes) else "unknown"
            conf = float(confidences[i])
            x1, y1, x2, y2 = [float(coord) for coord in boxes[i]]
            bbox = [round(coord, 1) for coord in boxes[i]]
            print(f"- {label} ({round(conf * 100, 1)}%) at {bbox}")

            updated_preds.append({
                "class": label,
                "confidence": conf,
                "bounding_box": {
                    "x": x1,
                    "y": y1,
                    "width": x2 - x1,
                    "height": y2 - y1
                }
            })

    with lock:
        latest_predictions = updated_preds

def display_loop():
    global running, frame_count, fps, last_fps_time

    try:
        while running:
            with lock:
                frame = latest_frame.copy() if latest_frame is not None else None
                predictions = list(latest_predictions)

            if frame is not None:
                # Draw predictions
                for pred in predictions:
                    bbox = pred.get("bounding_box", {})
                    label = pred.get("class", "unknown")
                    conf = round(pred.get("confidence", 0) * 100, 1)

                    x1 = int(bbox.get("x", 0))
                    y1 = int(bbox.get("y", 0))
                    w = int(bbox.get("width", 0))
                    h = int(bbox.get("height", 0))
                    x2, y2 = x1 + w, y1 + h

                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label_text = f"{label} ({conf}%)"
                    cv2.putText(frame, label_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Calculate and display FPS
                frame_count += 1
                current_time = time.time()
                if current_time - last_fps_time >= 1.0:
                    fps = frame_count / (current_time - last_fps_time)
                    last_fps_time = current_time
                    frame_count = 0

                cv2.putText(frame, f"FPS: {fps:.2f}", (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

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
    except Exception as e:
        logging.error(f"Error in pipeline thread: {e}")

def main():
    global running

    try:
        # Show available providers
        providers = ort.get_available_providers()
        logging.info(f"Available ONNX Runtime providers: {providers}")
        if "CUDAExecutionProvider" not in providers:
            logging.warning("CUDAExecutionProvider not found. Using CPUExecutionProvider.")

        # Initialize inference pipeline
        pipeline = InferencePipeline.init_with_workflow(
            api_key=API_KEY,
            workspace_name="adaptive-traffic-light-v2",
            workflow_id="detect-and-classify-2",
            video_reference="http://10.68.14.208:8080/video",
            # video_reference=0,
            max_fps=60,
            on_prediction=my_sink
        )

        # Start inference and display threads
        pipeline_thread = threading.Thread(target=run_pipeline, args=(pipeline,))
        display_thread = threading.Thread(target=display_loop)

        pipeline_thread.start()
        display_thread.start()

        display_thread.join()
        running = False

        # Stop the pipeline if supported
        if hasattr(pipeline, "stop"):
            pipeline.stop()
            logging.info("Pipeline stopped.")

        pipeline_thread.join()
        logging.info("Shutdown complete.")

    except KeyboardInterrupt:
        logging.info("KeyboardInterrupt received. Exiting...")
        running = False
        if 'pipeline' in locals() and hasattr(pipeline, "stop"):
            pipeline.stop()
    except Exception as e:
        logging.error(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()
