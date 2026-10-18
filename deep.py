import streamlit as st
from pathlib import Path
import tempfile
import subprocess
import cv2
import numpy as np
import onnxruntime as ort

from insightface.app import FaceAnalysis
from insightface.model_zoo import get_model


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Synthetic Media Generator",
    page_icon="🎭",
    layout="wide"
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "models"

GENDER_MODEL_PATH = MODEL_DIR / "gender_net.onnx"
SWAP_MODEL_PATH = MODEL_DIR / "inswapper_128.onnx"

WATERMARK_TEXT = "AI-GENERATED / SYNTHETIC MEDIA"

GENDER_CONFIDENCE_THRESHOLD = 0.70

# Change these only if your gender model uses the opposite order.
GENDER_INDEX = {
    "female": 0,
    "male": 1
}


# ============================================================
# CHECK MODELS
# ============================================================

def check_models():

    missing = []

    if not GENDER_MODEL_PATH.exists():
        missing.append(
            f"gender_net.onnx\n{GENDER_MODEL_PATH}"
        )

    if not SWAP_MODEL_PATH.exists():
        missing.append(
            f"inswapper_128.onnx\n{SWAP_MODEL_PATH}"
        )

    if missing:

        message = (
            "Required model file(s) are missing:\n\n"
            + "\n\n".join(missing)
            + "\n\nExpected structure:\n"
            f"{MODEL_DIR}/\n"
            "    gender_net.onnx\n"
            "    inswapper_128.onnx"
        )

        raise FileNotFoundError(message)


# ============================================================
# LOAD INSIGHTFACE + SWAPPER
# ============================================================

@st.cache_resource(show_spinner=False)
def load_face_models():

    if not SWAP_MODEL_PATH.exists():

        raise FileNotFoundError(
            "inswapper_128.onnx was not found:\n"
            f"{SWAP_MODEL_PATH}"
        )

    try:

        app = FaceAnalysis(
            name="buffalo_l",
            providers=[
                "CPUExecutionProvider"
            ]
        )

        app.prepare(
            ctx_id=0,
            det_size=(640, 640)
        )

    except Exception as error:

        raise RuntimeError(
            "Could not initialize InsightFace.\n\n"
            f"{error}"
        )

    try:

        swapper = get_model(
            str(SWAP_MODEL_PATH),
            providers=[
                "CPUExecutionProvider"
            ]
        )

    except Exception as error:

        raise RuntimeError(
            "Could not load inswapper_128.onnx.\n\n"
            f"Model path:\n{SWAP_MODEL_PATH}\n\n"
            f"Error:\n{error}"
        )

    return app, swapper


# ============================================================
# LOAD GENDER MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_gender_model():

    if not GENDER_MODEL_PATH.exists():

        raise FileNotFoundError(
            "gender_net.onnx was not found:\n"
            f"{GENDER_MODEL_PATH}"
        )

    try:

        session = ort.InferenceSession(
            str(GENDER_MODEL_PATH),
            providers=[
                "CPUExecutionProvider"
            ]
        )

    except Exception as error:

        raise RuntimeError(
            "Could not load gender_net.onnx using "
            "ONNX Runtime.\n\n"
            "The file may be corrupted, incomplete, "
            "or may not be a valid ONNX model.\n\n"
            f"Path:\n{GENDER_MODEL_PATH}\n\n"
            f"Error:\n{error}"
        )

    inputs = session.get_inputs()
    outputs = session.get_outputs()

    if not inputs:

        raise RuntimeError(
            "Gender model has no input."
        )

    if not outputs:

        raise RuntimeError(
            "Gender model has no output."
        )

    input_info = inputs[0]

    output_info = outputs[0]

    return (
        session,
        input_info.name,
        output_info.name,
        input_info.shape
    )


# ============================================================
# GENDER CLASSIFIER
# ============================================================

class GenderClassifier:

    def __init__(self):

        (
            self.session,
            self.input_name,
            self.output_name,
            self.input_shape
        ) = load_gender_model()

    # --------------------------------------------------------
    # Crop face
    # --------------------------------------------------------

    def crop_face(
        self,
        image,
        bbox
    ):

        x1, y1, x2, y2 = bbox

        h, w = image.shape[:2]

        x1 = max(
            0,
            int(x1)
        )

        y1 = max(
            0,
            int(y1)
        )

        x2 = min(
            w,
            int(x2)
        )

        y2 = min(
            h,
            int(y2)
        )

        if x2 <= x1 or y2 <= y1:

            return None

        face = image[
            y1:y2,
            x1:x2
        ]

        if face.size == 0:

            return None

        return face

    # --------------------------------------------------------
    # Prepare input
    # --------------------------------------------------------

    def prepare_input(
        self,
        face
    ):

        # Most common gender ONNX models use 224x224.
        # Change this only if your model reports another size.

        input_height = 224
        input_width = 224

        # BGR -> RGB
        face = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB
        )

        # Resize
        face = cv2.resize(
            face,
            (
                input_width,
                input_height
            ),
            interpolation=cv2.INTER_AREA
        )

        face = face.astype(
            np.float32
        )

        # 0-1
        face /= 255.0

        # Normalize to -1 to +1
        face = (
            face - 0.5
        ) / 0.5

        # HWC -> CHW
        face = np.transpose(
            face,
            (2, 0, 1)
        )

        # Add batch dimension
        blob = np.expand_dims(
            face,
            axis=0
        )

        blob = np.ascontiguousarray(
            blob,
            dtype=np.float32
        )

        return blob

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    def predict(
        self,
        image,
        bbox
    ):

        face = self.crop_face(
            image,
            bbox
        )

        if face is None:

            return "unknown", 0.0

        try:

            blob = self.prepare_input(
                face
            )

            output = self.session.run(
                [self.output_name],
                {
                    self.input_name: blob
                }
            )[0]

        except Exception as error:

            print(
                f"Gender inference error: {error}"
            )

            return "unknown", 0.0

        scores = np.asarray(
            output,
            dtype=np.float32
        ).flatten()

        if len(scores) < 2:

            return "unknown", 0.0

        values = scores[:2]

        # ----------------------------------------------------
        # Convert logits to probabilities
        # ----------------------------------------------------

        if (
            np.any(values < 0.0)
            or
            np.any(values > 1.0)
            or
            abs(
                float(values.sum()) - 1.0
            ) > 0.10
        ):

            values = (
                values -
                np.max(values)
            )

            values = np.exp(values)

            total = np.sum(values)

            if total > 0:

                values /= total

        female_probability = float(
            values[
                GENDER_INDEX["female"]
            ]
        )

        male_probability = float(
            values[
                GENDER_INDEX["male"]
            ]
        )

        if (
            male_probability >=
            female_probability
        ):

            gender = "male"

            confidence = (
                male_probability
            )

        else:

            gender = "female"

            confidence = (
                female_probability
            )

        if (
            confidence <
            GENDER_CONFIDENCE_THRESHOLD
        ):

            return "unknown", confidence

        return gender, confidence


# ============================================================
# GET SOURCE FACE
# ============================================================

def get_largest_source_face(
    app,
    image
):

    faces = app.get(
        image
    )

    if not faces:

        raise ValueError(
            "No face was detected in the source image."
        )

    largest_face = max(
        faces,
        key=lambda face:
        (
            face.bbox[2] -
            face.bbox[0]
        )
        *
        (
            face.bbox[3] -
            face.bbox[1]
        )
    )

    return largest_face


# ============================================================
# SWAP FACE
# ============================================================

def swap_face(
    swapper,
    frame,
    target_face,
    source_face
):

    try:

        result = swapper.get(
            frame,
            target_face,
            source_face,
            paste_back=True
        )

        return result

    except Exception as error:

        print(
            f"Face swap error: {error}"
        )

        return frame


# ============================================================
# WATERMARK
# ============================================================

def add_watermark(
    frame
):

    h, w = frame.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX

    scale = max(
        0.45,
        min(w, h) / 1200
    )

    thickness = 2

    (
        text_width,
        text_height
    ), baseline = cv2.getTextSize(
        WATERMARK_TEXT,
        font,
        scale,
        thickness
    )

    margin = max(
        10,
        w // 100
    )

    x = margin

    y = (
        margin +
        text_height
    )

    # Background
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

    # Text
    cv2.putText(
        frame,
        WATERMARK_TEXT,
        (
            x,
            y
        ),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )

    return frame


# ============================================================
# PROCESS VIDEO
# ============================================================

def process_video(
    input_video,
    output_video,
    app,
    swapper,
    classifier,
    male_source_face,
    female_source_face,
    progress_callback=None
):

    cap = cv2.VideoCapture(
        str(input_video)
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open target video."
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

        fps = 25.0

    if width <= 0 or height <= 0:

        cap.release()

        raise RuntimeError(
            "Invalid video dimensions."
        )

    # --------------------------------------------------------
    # Writer
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_video),
        fourcc,
        fps,
        (
            width,
            height
        )
    )

    if not writer.isOpened():

        cap.release()

        raise RuntimeError(
            "Could not create temporary video."
        )

    frame_number = 0

    try:

        while True:

            success, frame = cap.read()

            if not success:

                break

            # ------------------------------------------------
            # Detect target faces
            # ------------------------------------------------

            faces = app.get(
                frame
            )

            result = frame.copy()

            # ------------------------------------------------
            # Process every face
            # ------------------------------------------------

            for target_face in faces:

                bbox = (
                    target_face.bbox
                )

                gender, confidence = (
                    classifier.predict(
                        frame,
                        bbox
                    )
                )

                # --------------------------------------------
                # Male target
                # --------------------------------------------

                if gender == "male":

                    result = swap_face(
                        swapper,
                        result,
                        target_face,
                        male_source_face
                    )

                # --------------------------------------------
                # Female target
                # --------------------------------------------

                elif gender == "female":

                    result = swap_face(
                        swapper,
                        result,
                        target_face,
                        female_source_face
                    )

                # --------------------------------------------
                # Unknown
                # --------------------------------------------

                else:

                    pass

            # ------------------------------------------------
            # Add disclosure
            # ------------------------------------------------

            result = add_watermark(
                result
            )

            writer.write(
                result
            )

            frame_number += 1

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            progress = (
                frame_number /
                max(
                    total_frames,
                    1
                )
            )

            if progress_callback:

                progress_callback(
                    progress
                )

    finally:

        cap.release()
        writer.release()

    return output_video


# ============================================================
# RESTORE ORIGINAL AUDIO
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

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:

        raise RuntimeError(
            "FFmpeg failed:\n\n"
            +
            result.stderr[-4000:]
        )

    return final_output


# ============================================================
# COMPLETE PIPELINE
# ============================================================

def create_synthetic_video(
    male_file,
    female_file,
    target_file,
    app,
    swapper,
    classifier,
    progress_callback=None
):

    with tempfile.TemporaryDirectory() as temp_dir:

        temp_dir = Path(
            temp_dir
        )

        # ----------------------------------------------------
        # Temporary files
        # ----------------------------------------------------

        male_path = (
            temp_dir /
            "male_source.jpg"
        )

        female_path = (
            temp_dir /
            "female_source.jpg"
        )

        target_path = (
            temp_dir /
            "target.mp4"
        )

        processed_path = (
            temp_dir /
            "processed.mp4"
        )

        final_path = (
            temp_dir /
            "synthetic_video.mp4"
        )

        # ----------------------------------------------------
        # Save uploaded files
        # ----------------------------------------------------

        with open(
            male_path,
            "wb"
        ) as file:

            file.write(
                male_file.getbuffer()
            )

        with open(
            female_path,
            "wb"
        ) as file:

            file.write(
                female_file.getbuffer()
            )

        with open(
            target_path,
            "wb"
        ) as file:

            file.write(
                target_file.getbuffer()
            )

        # ----------------------------------------------------
        # Read source images
        # ----------------------------------------------------

        male_image = cv2.imread(
            str(male_path)
        )

        female_image = cv2.imread(
            str(female_path)
        )

        if male_image is None:

            raise ValueError(
                "Could not read male source image."
            )

        if female_image is None:

            raise ValueError(
                "Could not read female source image."
            )

        # ----------------------------------------------------
        # Find male source face
        # ----------------------------------------------------

        male_source_face = (
            get_largest_source_face(
                app,
                male_image
            )
        )

        # ----------------------------------------------------
        # Find female source face
        # ----------------------------------------------------

        female_source_face = (
            get_largest_source_face(
                app,
                female_image
            )
        )

        # ----------------------------------------------------
        # Process video
        # ----------------------------------------------------

        process_video(

            target_path,

            processed_path,

            app,

            swapper,

            classifier,

            male_source_face,

            female_source_face,

            progress_callback
        )

        # ----------------------------------------------------
        # Restore audio
        # ----------------------------------------------------

        combine_video_audio(

            processed_path,

            target_path,

            final_path
        )

        # ----------------------------------------------------
        # Read result
        # ----------------------------------------------------

        with open(
            final_path,
            "rb"
        ) as file:

            output_data = file.read()

    return output_data


# ============================================================
# STREAMLIT UI
# ============================================================

st.title(
    "🎭 AI Synthetic Media Generator"
)

st.write(
    "Authorized synthetic-media demonstration"
)

st.warning(
    "Use only images and videos for which you have "
    "permission. The generated video is visibly labeled "
    "as AI-generated synthetic media."
)


# ============================================================
# MODEL STATUS
# ============================================================

st.subheader(
    "🧠 Model Status"
)

gender_exists = (
    GENDER_MODEL_PATH.exists()
)

swap_exists = (
    SWAP_MODEL_PATH.exists()
)

col1, col2 = st.columns(2)

with col1:

    if gender_exists:

        st.success(
            "✅ gender_net.onnx found"
        )

    else:

        st.error(
            "❌ gender_net.onnx missing"
        )

with col2:

    if swap_exists:

        st.success(
            "✅ inswapper_128.onnx found"
        )

    else:

        st.error(
            "❌ inswapper_128.onnx missing"
        )


if not gender_exists or not swap_exists:

    st.info(
        "Put both ONNX files inside the models folder."
    )

    st.code(
        str(MODEL_DIR)
    )

    st.stop()


# ============================================================
# UPLOAD SOURCE IMAGES
# ============================================================

st.subheader(
    "1️⃣ Source Images"
)

col1, col2 = st.columns(2)

with col1:

    male_file = st.file_uploader(
        "👨 Male source image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        key="male_source"
    )

with col2:

    female_file = st.file_uploader(
        "👩 Female source image",
        type=[
            "jpg",
            "jpeg",
            "png"
        ],
        key="female_source"
    )


# ============================================================
# TARGET VIDEO
# ============================================================

st.subheader(
    "2️⃣ Target Video"
)

target_file = st.file_uploader(
    "🎬 Upload target video",
    type=[
        "mp4",
        "mov",
        "avi",
        "mkv"
    ],
    key="target_video"
)


# ============================================================
# PREVIEW
# ============================================================

if male_file is not None:

    st.image(
        male_file,
        width=250,
        caption="Male source"
    )

if female_file is not None:

    st.image(
        female_file,
        width=250,
        caption="Female source"
    )

if target_file is not None:

    st.video(
        target_file
    )


# ============================================================
# GENERATE BUTTON
# ============================================================

generate = st.button(
    "🚀 Generate Synthetic Video",
    type="primary",
    use_container_width=True
)


# ============================================================
# GENERATION
# ============================================================

if generate:

    # --------------------------------------------------------
    # Validate uploads
    # --------------------------------------------------------

    if male_file is None:

        st.error(
            "Please upload the male source image."
        )

        st.stop()

    if female_file is None:

        st.error(
            "Please upload the female source image."
        )

        st.stop()

    if target_file is None:

        st.error(
            "Please upload the target video."
        )

        st.stop()

    progress = st.progress(
        0
    )

    status = st.empty()

    try:

        # ----------------------------------------------------
        # Load models
        # ----------------------------------------------------

        status.info(
            "Loading InsightFace and face-swap model..."
        )

        app, swapper = (
            load_face_models()
        )

        status.info(
            "Loading gender classifier..."
        )

        classifier = GenderClassifier()

        # ----------------------------------------------------
        # Progress callback
        # ----------------------------------------------------

        def update_progress(
            value
        ):

            value = max(
                0.0,
                min(
                    1.0,
                    float(value)
                )
            )

            progress.progress(
                value
            )

            status.info(
                f"Processing video: "
                f"{value * 100:.1f}%"
            )

        # ----------------------------------------------------
        # Generate
        # ----------------------------------------------------

        status.info(
            "Starting face processing..."
        )

        output_data = (
            create_synthetic_video(

                male_file,

                female_file,

                target_file,

                app,

                swapper,

                classifier,

                update_progress
            )
        )

        # ----------------------------------------------------
        # Finished
        # ----------------------------------------------------

        progress.progress(
            1.0
        )

        status.success(
            "✅ Synthetic video generated successfully."
        )

        st.subheader(
            "🎬 Output"
        )

        st.video(
            output_data
        )

        st.download_button(
            label="⬇️ Download Synthetic Video",
            data=output_data,
            file_name="synthetic_video.mp4",
            mime="video/mp4",
            use_container_width=True
        )

    except Exception as error:

        status.error(
            "❌ Processing failed."
        )

        st.exception(
            error
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI-GENERATED / SYNTHETIC MEDIA"
)
