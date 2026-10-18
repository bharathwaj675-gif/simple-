from pathlib import Path
import subprocess

import cv2
import numpy as np
import onnxruntime as ort

from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model


# ============================================================
# CONFIGURATION
# ============================================================

# Use the location of this Python file, NOT the current
# Jupyter/Streamlit working directory.
BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

SWAP_MODEL_PATH = MODEL_DIR / "inswapper_128.onnx"
GENDER_MODEL_PATH = MODEL_DIR / "gender_net.onnx"

MALE_SOURCE_PATH = INPUT_DIR / "male_source.jpg"
FEMALE_SOURCE_PATH = INPUT_DIR / "female_source.jpg"
TARGET_VIDEO_PATH = INPUT_DIR / "target.mp4"

FINAL_OUTPUT_PATH = OUTPUT_DIR / "synthetic_video.mp4"
TEMP_VIDEO_PATH = OUTPUT_DIR / "temp_no_audio.mp4"

GENDER_CONFIDENCE_THRESHOLD = 0.70

# Expected output:
# index 0 = female
# index 1 = male
GENDER_INDEX = {
    "female": 0,
    "male": 1
}

WATERMARK_TEXT = "AI-GENERATED / SYNTHETIC MEDIA"


# ============================================================
# CREATE DIRECTORIES
# ============================================================

MODEL_DIR.mkdir(parents=True, exist_ok=True)
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# MODEL CHECK
# ============================================================

def check_required_models():

    missing = []

    if not SWAP_MODEL_PATH.exists():
        missing.append(str(SWAP_MODEL_PATH))

    if not GENDER_MODEL_PATH.exists():
        missing.append(str(GENDER_MODEL_PATH))

    if missing:

        message = (
            "\nRequired model file(s) missing:\n\n"
            + "\n".join(missing)
            + "\n\nExpected structure:\n"
            f"{MODEL_DIR}/\n"
            "    inswapper_128.onnx\n"
            "    gender_net.onnx\n"
        )

        raise FileNotFoundError(message)

    print("\nModel files found:")

    print(
        f"inswapper_128.onnx: "
        f"{SWAP_MODEL_PATH.stat().st_size / (1024 * 1024):.2f} MB"
    )

    print(
        f"gender_net.onnx: "
        f"{GENDER_MODEL_PATH.stat().st_size / (1024 * 1024):.2f} MB"
    )


# ============================================================
# GENDER CLASSIFIER
# ============================================================

class GenderClassifier:

    def __init__(self, model_path=GENDER_MODEL_PATH):

        self.model_path = Path(model_path)

        if not self.model_path.exists():

            raise FileNotFoundError(
                f"Gender model not found:\n"
                f"{self.model_path}"
            )

        print("\nLoading gender model...")

        try:

            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=["CPUExecutionProvider"]
            )

        except Exception as error:

            raise RuntimeError(
                "\nCould not load gender_net.onnx with "
                "ONNX Runtime.\n\n"
                "The file may be corrupted, incomplete, "
                "or not actually be a valid ONNX model.\n\n"
                f"Model: {self.model_path}\n\n"
                f"Original error:\n{error}"
            )

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()

        if not inputs:

            raise RuntimeError(
                "Gender ONNX model has no input."
            )

        if not outputs:

            raise RuntimeError(
                "Gender ONNX model has no output."
            )

        self.input_name = inputs[0].name
        self.output_name = outputs[0].name

        self.input_shape = inputs[0].shape

        print("Gender model loaded.")

        print(
            f"Gender input: {self.input_name}"
        )

        print(
            f"Gender output: {self.output_name}"
        )

        print(
            f"Gender input shape: {self.input_shape}"
        )


    # ========================================================
    # CROP FACE
    # ========================================================

    def crop_face(self, image, face):

        h, w = image.shape[:2]

        x1, y1, x2, y2 = face.bbox.astype(int)

        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:

            return None

        crop = image[y1:y2, x1:x2]

        if crop.size == 0:

            return None

        return crop


    # ========================================================
    # PREPARE INPUT
    # ========================================================

    def prepare_input(self, crop):

        blob = cv2.dnn.blobFromImage(
            crop,
            scalefactor=1.0,
            size=(224, 224),
            mean=(104, 117, 123),
            swapRB=False,
            crop=False
        )

        return blob.astype(np.float32)


    # ========================================================
    # PREDICT GENDER
    # ========================================================

    def predict(self, image, face):

        crop = self.crop_face(
            image,
            face
        )

        if crop is None:

            return "unknown", 0.0

        try:

            blob = self.prepare_input(crop)

            prediction = self.session.run(
                [self.output_name],
                {
                    self.input_name: blob
                }
            )[0]

        except Exception as error:

            print(
                f"\nGender inference error: {error}"
            )

            return "unknown", 0.0

        prediction = np.asarray(
            prediction,
            dtype=np.float32
        ).flatten()

        if len(prediction) < 2:

            print(
                "\nWARNING: Gender model returned "
                f"{len(prediction)} output values."
            )

            return "unknown", 0.0

        # ----------------------------------------------------
        # Get two class scores
        # ----------------------------------------------------

        female_score = float(
            prediction[GENDER_INDEX["female"]]
        )

        male_score = float(
            prediction[GENDER_INDEX["male"]]
        )

        scores = np.array(
            [
                female_score,
                male_score
            ],
            dtype=np.float32
        )

        # ----------------------------------------------------
        # Convert logits to probabilities if necessary
        # ----------------------------------------------------

        if (
            np.any(scores < 0.0)
            or np.any(scores > 1.0)
            or abs(float(scores.sum()) - 1.0) > 0.10
        ):

            shifted = (
                scores -
                np.max(scores)
            )

            exp_scores = np.exp(shifted)

            probabilities = (
                exp_scores /
                np.sum(exp_scores)
            )

            female_score = float(
                probabilities[0]
            )

            male_score = float(
                probabilities[1]
            )

        # ----------------------------------------------------
        # Select gender
        # ----------------------------------------------------

        if male_score >= female_score:

            gender = "male"
            confidence = male_score

        else:

            gender = "female"
            confidence = female_score

        # ----------------------------------------------------
        # Confidence check
        # ----------------------------------------------------

        if confidence < GENDER_CONFIDENCE_THRESHOLD:

            return "unknown", confidence

        return gender, confidence


# ============================================================
# FACE SWAPPER
# ============================================================

class FaceSwapper:

    def __init__(self, model_path=SWAP_MODEL_PATH):

        self.model_path = Path(model_path)

        if not self.model_path.exists():

            raise FileNotFoundError(
                f"Face swap model not found:\n"
                f"{self.model_path}"
            )

        print("\nLoading InsightFace...")

        # ----------------------------------------------------
        # Face Analysis
        # ----------------------------------------------------

        self.app = FaceAnalysis(
            name="buffalo_l",
            providers=[
                "CPUExecutionProvider"
            ]
        )

        self.app.prepare(
            ctx_id=0,
            det_size=(640, 640)
        )

        print("Face analysis loaded.")

        # ----------------------------------------------------
        # Face Swap Model
        # ----------------------------------------------------

        print("Loading face swap model...")

        try:

            self.swapper = get_model(
                str(self.model_path),
                providers=[
                    "CPUExecutionProvider"
                ]
            )

        except Exception as error:

            raise RuntimeError(
                "\nCould not load inswapper_128.onnx.\n\n"
                f"Model: {self.model_path}\n\n"
                f"Original error:\n{error}"
            )

        print("Face swap model loaded.")


    # ========================================================
    # GET LARGEST FACE
    # ========================================================

    def get_largest_face(self, image):

        faces = self.app.get(image)

        if len(faces) == 0:

            raise ValueError(
                "No face found in source image."
            )

        face = max(
            faces,
            key=lambda f:
                (
                    f.bbox[2] - f.bbox[0]
                )
                *
                (
                    f.bbox[3] - f.bbox[1]
                )
        )

        return face


    # ========================================================
    # SWAP SINGLE FACE
    # ========================================================

    def swap_single_face(
        self,
        image,
        target_face,
        source_face
    ):

        return self.swapper.get(
            image,
            target_face,
            source_face,
            paste_back=True
        )


# ============================================================
# WATERMARK
# ============================================================

def add_watermark(
    frame,
    text=WATERMARK_TEXT
):

    height, width = frame.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX

    scale = max(
        0.45,
        min(width, height) / 1200
    )

    thickness = 2

    (
        text_width,
        text_height
    ), baseline = cv2.getTextSize(
        text,
        font,
        scale,
        thickness
    )

    margin = max(
        10,
        width // 100
    )

    x = margin

    y = margin + text_height

    # --------------------------------------------------------
    # Background
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (
            x - 6,
            y - text_height - 6
        ),
        (
            x + text_width + 6,
            y + baseline + 6
        ),
        (20, 20, 20),
        -1
    )

    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    cv2.putText(
        frame,
        text,
        (x, y),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )

    return frame


# ============================================================
# VIDEO PROCESSING
# ============================================================

def process_video(
    input_video,
    output_video,
    face_swapper,
    gender_classifier,
    male_source_face,
    female_source_face,
    progress_callback=None
):

    input_video = str(input_video)
    output_video = str(output_video)

    cap = cv2.VideoCapture(input_video)

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open input video."
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if fps <= 0:

        fps = 25

    if width <= 0 or height <= 0:

        cap.release()

        raise RuntimeError(
            "Invalid video dimensions."
        )

    # --------------------------------------------------------
    # Video writer
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        output_video,
        fourcc,
        fps,
        (width, height)
    )

    if not writer.isOpened():

        cap.release()

        raise RuntimeError(
            "Could not create output video."
        )

    frame_number = 0

    print("\nProcessing video...")

    try:

        while True:

            success, frame = cap.read()

            if not success:

                break

            # ------------------------------------------------
            # Detect faces
            # ------------------------------------------------

            faces = face_swapper.app.get(
                frame
            )

            result = frame.copy()

            # ------------------------------------------------
            # Process detected faces
            # ------------------------------------------------

            for target_face in faces:

                gender, confidence = (
                    gender_classifier.predict(
                        frame,
                        target_face
                    )
                )

                if gender == "male":

                    result = (
                        face_swapper.swap_single_face(
                            result,
                            target_face,
                            male_source_face
                        )
                    )

                elif gender == "female":

                    result = (
                        face_swapper.swap_single_face(
                            result,
                            target_face,
                            female_source_face
                        )
                    )

            # ------------------------------------------------
            # Add disclosure watermark
            # ------------------------------------------------

            result = add_watermark(
                result
            )

            writer.write(result)

            frame_number += 1

            progress = (
                frame_number /
                max(total_frames, 1)
            )

            if progress_callback:

                progress_callback(progress)

            else:

                print(
                    f"\rProgress: "
                    f"{progress * 100:6.2f}%",
                    end=""
                )

    finally:

        cap.release()
        writer.release()

    print(
        "\n\nVideo processing completed."
    )

    return output_video


# ============================================================
# AUDIO + VIDEO
# ============================================================

def combine_video_audio(
    processed_video,
    original_video,
    final_output
):

    command = [

        "ffmpeg",

        "-y",

        "-i",
        str(processed_video),

        "-i",
        str(original_video),

        "-map",
        "0:v:0",

        "-map",
        "1:a:0?",

        "-c:v",
        "libx264",

        "-preset",
        "veryfast",

        "-crf",
        "23",

        "-c:a",
        "aac",

        "-shortest",

        str(final_output)
    ]

    print(
        "\nCombining original audio..."
    )

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:

        raise RuntimeError(
            "FFmpeg failed:\n\n"
            + result.stderr[-3000:]
        )

    print(
        "Audio successfully restored."
    )

    return str(final_output)


# ============================================================
# COMPLETE VIDEO PIPELINE
# ============================================================

def create_synthetic_video(
    male_source_path,
    female_source_path,
    target_video_path,
    progress_callback=None
):

    male_source_path = Path(
        male_source_path
    )

    female_source_path = Path(
        female_source_path
    )

    target_video_path = Path(
        target_video_path
    )

    # --------------------------------------------------------
    # Validate input files
    # --------------------------------------------------------

    if not male_source_path.exists():

        raise FileNotFoundError(
            f"Male source image not found:\n"
            f"{male_source_path}"
        )

    if not female_source_path.exists():

        raise FileNotFoundError(
            f"Female source image not found:\n"
            f"{female_source_path}"
        )

    if not target_video_path.exists():

        raise FileNotFoundError(
            f"Target video not found:\n"
            f"{target_video_path}"
        )

    # --------------------------------------------------------
    # Check models
    # --------------------------------------------------------

    check_required_models()

    # --------------------------------------------------------
    # Load models ONCE
    # --------------------------------------------------------

    face_swapper = FaceSwapper(
        SWAP_MODEL_PATH
    )

    gender_classifier = GenderClassifier(
        GENDER_MODEL_PATH
    )

    # --------------------------------------------------------
    # Read male source
    # --------------------------------------------------------

    male_image = cv2.imread(
        str(male_source_path)
    )

    if male_image is None:

        raise ValueError(
            "Could not read male source image."
        )

    # --------------------------------------------------------
    # Read female source
    # --------------------------------------------------------

    female_image = cv2.imread(
        str(female_source_path)
    )

    if female_image is None:

        raise ValueError(
            "Could not read female source image."
        )

    # --------------------------------------------------------
    # Detect source faces
    # --------------------------------------------------------

    print(
        "\nDetecting male source face..."
    )

    male_source_face = (
        face_swapper.get_largest_face(
            male_image
        )
    )

    print(
        "Detecting female source face..."
    )

    female_source_face = (
        face_swapper.get_largest_face(
            female_image
        )
    )

    # --------------------------------------------------------
    # Delete old temporary output
    # --------------------------------------------------------

    if TEMP_VIDEO_PATH.exists():

        TEMP_VIDEO_PATH.unlink()

    if FINAL_OUTPUT_PATH.exists():

        FINAL_OUTPUT_PATH.unlink()

    # --------------------------------------------------------
    # Process video
    # --------------------------------------------------------

    process_video(

        target_video_path,

        TEMP_VIDEO_PATH,

        face_swapper,

        gender_classifier,

        male_source_face,

        female_source_face,

        progress_callback
    )

    # --------------------------------------------------------
    # Restore original audio
    # --------------------------------------------------------

    combine_video_audio(

        TEMP_VIDEO_PATH,

        target_video_path,

        FINAL_OUTPUT_PATH
    )

    # --------------------------------------------------------
    # Remove temporary video
    # --------------------------------------------------------

    if TEMP_VIDEO_PATH.exists():

        TEMP_VIDEO_PATH.unlink()

    # --------------------------------------------------------
    # Verify final output
    # --------------------------------------------------------

    if not FINAL_OUTPUT_PATH.exists():

        raise RuntimeError(
            "Final video was not created."
        )

    print(
        "\nFinal video:"
    )

    print(
        FINAL_OUTPUT_PATH
    )

    return str(
        FINAL_OUTPUT_PATH
    )


# ============================================================
# PROGRESS
# ============================================================

def show_progress(progress):

    percentage = progress * 100

    print(
        f"\rProcessing: "
        f"{percentage:6.2f}%",
        end=""
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)

    print(
        "AI SYNTHETIC MEDIA GENERATOR"
    )

    print(
        "MALE → MALE | FEMALE → FEMALE"
    )

    print("=" * 60)

    print(
        f"\nProject directory:\n{BASE_DIR}"
    )

    try:

        output = create_synthetic_video(

            MALE_SOURCE_PATH,

            FEMALE_SOURCE_PATH,

            TARGET_VIDEO_PATH,

            show_progress
        )

        print("\n")

        print("=" * 60)

        print(
            "PROCESS COMPLETED"
        )

        print("=" * 60)

        print(
            f"\nOutput:\n{output}"
        )

    except Exception as error:

        print("\n")

        print("=" * 60)

        print(
            "ERROR"
        )

        print("=" * 60)

        print(
            f"\n{error}"
        )
