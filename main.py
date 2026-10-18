from pathlib import Path
import subprocess
import cv2
import numpy as np

from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path.cwd()

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

# ------------------------------------------------------------
# Gender settings
# ------------------------------------------------------------

# Do not swap if the classifier is not confident enough.
GENDER_CONFIDENCE_THRESHOLD = 0.70

# Expected gender model output:
#
# index 0 = female
# index 1 = male
#
# If your particular model uses the opposite order,
# change this to:
#
# GENDER_INDEX = {
#     "male": 0,
#     "female": 1
# }
#
GENDER_INDEX = {
    "female": 0,
    "male": 1
}

WATERMARK_TEXT = "AI-GENERATED / SYNTHETIC MEDIA"


# ============================================================
# CREATE DIRECTORIES
# ============================================================

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

INPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# GENDER CLASSIFIER
# ============================================================

class GenderClassifier:

    def __init__(
        self,
        model_path=GENDER_MODEL_PATH
    ):

        model_path = Path(model_path)

        if not model_path.exists():

            raise FileNotFoundError(
                f"\nGender model not found:\n"
                f"{model_path}"
            )

        self.net = cv2.dnn.readNetFromONNX(
            str(model_path)
        )

        print(
            "Gender classification model loaded."
        )


    # ========================================================
    # CROP FACE
    # ========================================================

    def crop_face(
        self,
        image,
        face
    ):

        h, w = image.shape[:2]

        x1, y1, x2, y2 = face.bbox.astype(int)

        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:

            return None

        crop = image[
            y1:y2,
            x1:x2
        ]

        if crop.size == 0:

            return None

        return crop


    # ========================================================
    # PREDICT GENDER
    # ========================================================

    def predict(
        self,
        image,
        face
    ):

        crop = self.crop_face(
            image,
            face
        )

        if crop is None:

            return "unknown", 0.0


        # ----------------------------------------------------
        # Prepare image
        # ----------------------------------------------------

        blob = cv2.dnn.blobFromImage(
            crop,
            scalefactor=1.0,
            size=(224, 224),
            mean=(104, 117, 123),
            swapRB=False,
            crop=False
        )

        self.net.setInput(blob)

        prediction = self.net.forward()

        prediction = np.asarray(
            prediction
        ).flatten()


        # ----------------------------------------------------
        # Check model output
        # ----------------------------------------------------

        if len(prediction) < 2:

            print(
                "\nWARNING: Gender model returned "
                "less than two outputs."
            )

            return "unknown", 0.0


        # ----------------------------------------------------
        # Get class scores
        # ----------------------------------------------------

        female_score = float(
            prediction[
                GENDER_INDEX["female"]
            ]
        )

        male_score = float(
            prediction[
                GENDER_INDEX["male"]
            ]
        )


        # ----------------------------------------------------
        # Convert to probabilities if required
        # ----------------------------------------------------
        #
        # Some models return logits rather than probabilities.
        # If values don't look like probabilities, use softmax.
        #

        scores = np.array(
            [
                female_score,
                male_score
            ],
            dtype=np.float32
        )

        if (
            np.any(scores < 0.0)
            or np.any(scores > 1.0)
            or abs(float(scores.sum()) - 1.0) > 0.10
        ):

            exp_scores = np.exp(
                scores - np.max(scores)
            )

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

        if male_score > female_score:

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

    def __init__(
        self,
        model_path=SWAP_MODEL_PATH
    ):

        model_path = Path(
            model_path
        )

        if not model_path.exists():

            raise FileNotFoundError(
                f"\nFace swap model not found:\n"
                f"{model_path}"
            )


        print(
            "\nLoading InsightFace..."
        )


        # ----------------------------------------------------
        # Face detection
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


        print(
            "Face analysis loaded."
        )


        # ----------------------------------------------------
        # Face swap model
        # ----------------------------------------------------

        print(
            "Loading face swap model..."
        )

        self.swapper = get_model(
            str(model_path),
            providers=[
                "CPUExecutionProvider"
            ]
        )


        print(
            "Face swap model loaded."
        )


    # ========================================================
    # GET LARGEST FACE
    # ========================================================

    def get_largest_face(
        self,
        image
    ):

        faces = self.app.get(
            image
        )

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

    input_video = str(
        input_video
    )

    output_video = str(
        output_video
    )


    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        input_video
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open input video."
        )


    # --------------------------------------------------------
    # Video information
    # --------------------------------------------------------

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

    print(
        "\nProcessing video..."
    )


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
            # Process every detected face
            # ------------------------------------------------

            for target_face in faces:

                gender, confidence = (
                    gender_classifier.predict(
                        frame,
                        target_face
                    )
                )


                # ============================================
                # MALE → MALE
                # ============================================

                if gender == "male":

                    result = (
                        face_swapper.swap_single_face(
                            result,
                            target_face,
                            male_source_face
                        )
                    )


                # ============================================
                # FEMALE → FEMALE
                # ============================================

                elif gender == "female":

                    result = (
                        face_swapper.swap_single_face(
                            result,
                            target_face,
                            female_source_face
                        )
                    )


                # ============================================
                # UNKNOWN → NO SWAP
                # ============================================

                else:

                    continue


            # ------------------------------------------------
            # Mandatory disclosure
            # ------------------------------------------------

            result = add_watermark(
                result
            )


            # ------------------------------------------------
            # Write frame
            # ------------------------------------------------

            writer.write(
                result
            )

            frame_number += 1


            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            progress = (
                frame_number /
                max(total_frames, 1)
            )


            if progress_callback:

                progress_callback(
                    progress
                )

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


    return str(
        final_output
    )


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
    # Validate files
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
    # Load models
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
    # Temporary output
    # --------------------------------------------------------

    if TEMP_VIDEO_PATH.exists():

        TEMP_VIDEO_PATH.unlink()


    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

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
    # Remove temporary file
    # --------------------------------------------------------

    if TEMP_VIDEO_PATH.exists():

        TEMP_VIDEO_PATH.unlink()


    # --------------------------------------------------------
    # Verify
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

def show_progress(
    progress
):

    percentage = (
        progress * 100
    )

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