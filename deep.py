import streamlit as st
from pathlib import Path
import tempfile
import subprocess
import cv2
import numpy as np
import time


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Synthetic Media Generator",
    page_icon="🎭",
    layout="wide"
)


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "models" / "gender_net.onnx"

WATERMARK_TEXT = "AI-GENERATED / SYNTHETIC MEDIA"

GENDER_CONFIDENCE_THRESHOLD = 0.70

GENDER_INDEX = {
    "female": 0,
    "male": 1
}


# ============================================================
# FACE DETECTOR
# ============================================================

class FaceDetector:

    def __init__(self):

        cascade_path = (
            cv2.data.haarcascades
            + "haarcascade_frontalface_default.xml"
        )

        self.detector = cv2.CascadeClassifier(
            cascade_path
        )

        if self.detector.empty():

            raise RuntimeError(
                "Could not load Haar Cascade."
            )

    def detect(self, image):

        if image is None:
            return []

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.equalizeHist(gray)

        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(50, 50)
        )

        result = []

        for x, y, w, h in faces:

            result.append(
                (
                    int(x),
                    int(y),
                    int(x + w),
                    int(y + h)
                )
            )

        return result

    def largest(self, image):

        faces = self.detect(image)

        if not faces:

            raise ValueError(
                "No face detected."
            )

        return max(
            faces,
            key=lambda b:
            (b[2] - b[0])
            *
            (b[3] - b[1])
        )


# ============================================================
# GENDER CLASSIFIER
# ============================================================

class GenderClassifier:

    def __init__(self, model_path):

        model_path = Path(model_path)

        if not model_path.exists():

            raise FileNotFoundError(
                f"Model not found:\n{model_path}"
            )

        self.net = cv2.dnn.readNetFromONNX(
            str(model_path)
        )

    def crop_face(self, image, bbox):

        x1, y1, x2, y2 = bbox

        h, w = image.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:
            return None

        face = image[y1:y2, x1:x2]

        if face.size == 0:
            return None

        return face

    def predict(self, image, bbox):

        face = self.crop_face(
            image,
            bbox
        )

        if face is None:

            return "unknown", 0.0

        # RGB
        face = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB
        )

        # 224 x 224
        face = cv2.resize(
            face,
            (224, 224),
            interpolation=cv2.INTER_AREA
        )

        # 0-1
        face = (
            face.astype(
                np.float32
            ) / 255.0
        )

        # Normalize
        face = (
            face - 0.5
        ) / 0.5

        # HWC -> CHW
        face = np.transpose(
            face,
            (2, 0, 1)
        )

        # Batch
        blob = np.expand_dims(
            face,
            axis=0
        )

        blob = np.ascontiguousarray(
            blob,
            dtype=np.float32
        )

        self.net.setInput(blob)

        output = self.net.forward()

        scores = np.asarray(
            output,
            dtype=np.float32
        ).flatten()

        if len(scores) < 2:

            return "unknown", 0.0

        values = scores[:2]

        # Convert logits to probabilities
        if (
            np.any(values < 0)
            or np.any(values > 1)
            or abs(
                float(values.sum()) - 1
            ) > 0.10
        ):

            values = (
                values
                - np.max(values)
            )

            values = np.exp(values)

            total = np.sum(values)

            if total > 0:

                values /= total

        female = float(values[0])
        male = float(values[1])

        if male > female:

            gender = "male"
            confidence = male

        else:

            gender = "female"
            confidence = female

        if confidence < GENDER_CONFIDENCE_THRESHOLD:

            return "unknown", confidence

        return gender, confidence


# ============================================================
# SYNTHETIC FACE TRANSFORMATION
# ============================================================

def apply_synthetic_effect(
    frame,
    bbox,
    gender
):

    x1, y1, x2, y2 = bbox

    face = frame[
        y1:y2,
        x1:x2
    ]

    if face.size == 0:
        return frame

    h, w = face.shape[:2]

    # Pixelation
    small_w = max(
        8,
        w // 12
    )

    small_h = max(
        8,
        h // 12
    )

    small = cv2.resize(
        face,
        (small_w, small_h),
        interpolation=cv2.INTER_AREA
    )

    pixelated = cv2.resize(
        small,
        (w, h),
        interpolation=cv2.INTER_NEAREST
    )

    # Elliptical mask
    mask = np.zeros(
        (h, w),
        dtype=np.uint8
    )

    center = (
        w // 2,
        h // 2
    )

    axes = (
        max(1, int(w * 0.45)),
        max(1, int(h * 0.48))
    )

    cv2.ellipse(
        mask,
        center,
        axes,
        0,
        0,
        360,
        255,
        -1
    )

    mask = cv2.GaussianBlur(
        mask,
        (21, 21),
        0
    )

    alpha = (
        mask.astype(
            np.float32
        ) / 255.0
    )

    alpha = alpha[:, :, None]

    blended = (
        pixelated.astype(np.float32)
        * alpha
        +
        face.astype(np.float32)
        * (1 - alpha)
    )

    blended = np.clip(
        blended,
        0,
        255
    ).astype(np.uint8)

    frame[
        y1:y2,
        x1:x2
    ] = blended

    return frame


# ============================================================
# WATERMARK
# ============================================================

def add_watermark(frame):

    h, w = frame.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX

    scale = max(
        0.45,
        min(w, h) / 1200
    )

    thickness = 2

    (tw, th), baseline = cv2.getTextSize(
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
    y = margin + th

    cv2.rectangle(
        frame,
        (
            x - 6,
            y - th - 6
        ),
        (
            x + tw + 6,
            y + baseline + 6
        ),
        (20, 20, 20),
        -1
    )

    cv2.putText(
        frame,
        WATERMARK_TEXT,
        (x, y),
        font,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )

    return frame


# ============================================================
# VIDEO PROCESSOR
# ============================================================

def process_video(
    input_video,
    output_video,
    detector,
    classifier,
    progress_callback=None
):

    cap = cv2.VideoCapture(
        str(input_video)
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Could not open video."
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

    total = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if fps <= 0:
        fps = 25

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_video),
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

    try:

        while True:

            ok, frame = cap.read()

            if not ok:
                break

            faces = detector.detect(
                frame
            )

            result = frame.copy()

            for bbox in faces:

                gender, confidence = (
                    classifier.predict(
                        frame,
                        bbox
                    )
                )

                if gender in (
                    "male",
                    "female"
                ):

                    result = (
                        apply_synthetic_effect(
                            result,
                            bbox,
                            gender
                        )
                    )

            result = add_watermark(
                result
            )

            writer.write(
                result
            )

            frame_number += 1

            if progress_callback:

                progress_callback(
                    frame_number
                    /
                    max(total, 1)
                )

    finally:

        cap.release()
        writer.release()

    return output_video


# ============================================================
# RESTORE AUDIO
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
            result.stderr[-3000:]
        )

    return final_output


# ============================================================
# BACKEND
# ============================================================

def create_synthetic_video(
    male_file,
    female_file,
    target_file,
    detector,
    classifier,
    progress_callback=None
):

    with tempfile.TemporaryDirectory() as temp_dir:

        temp = Path(temp_dir)

        male_path = (
            temp / "male.jpg"
        )

        female_path = (
            temp / "female.jpg"
        )

        target_path = (
            temp / "target.mp4"
        )

        processed_path = (
            temp / "processed.mp4"
        )

        final_path = (
            temp / "synthetic_video.mp4"
        )

        # Save uploads
        with open(
            male_path,
            "wb"
        ) as f:

            f.write(
                male_file.getbuffer()
            )

        with open(
            female_path,
            "wb"
        ) as f:

            f.write(
                female_file.getbuffer()
            )

        with open(
            target_path,
            "wb"
        ) as f:

            f.write(
                target_file.getbuffer()
            )

        male_image = cv2.imread(
            str(male_path)
        )

        female_image = cv2.imread(
            str(female_path)
        )

        if male_image is None:
            raise ValueError(
                "Could not read male image."
            )

        if female_image is None:
            raise ValueError(
                "Could not read female image."
            )

        process_video(
            target_path,
            processed_path,
            detector,
            classifier,
            progress_callback
        )

        combine_video_audio(
            processed_path,
            target_path,
            final_path
        )

        with open(
            final_path,
            "rb"
        ) as f:

            output = f.read()

    return output


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
    "Use only media for which you have permission. "
    "The output is visibly labeled as synthetic media."
)


# ============================================================
# MODEL
# ============================================================

st.header(
    "🧠 Gender Model"
)

if not MODEL_PATH.exists():

    st.error(
        "gender_net.onnx is missing."
    )

    st.code(
        str(MODEL_PATH)
    )

    st.stop()

st.success(
    "✅ gender_net.onnx found"
)


# ============================================================
# UPLOADS
# ============================================================

st.header(
    "1. Upload Source Images"
)

col1, col2 = st.columns(2)

with col1:

    male_file = st.file_uploader(
        "👨 Male Source",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )

with col2:

    female_file = st.file_uploader(
        "👩 Female Source",
        type=[
            "jpg",
            "jpeg",
            "png"
        ]
    )


st.header(
    "2. Upload Target Video"
)

target_file = st.file_uploader(
    "🎬 Target Video",
    type=[
        "mp4",
        "mov",
        "avi"
    ]
)


# ============================================================
# PREVIEW
# ============================================================

if male_file:

    st.image(
        male_file,
        width=250,
        caption="Male source"
    )

if female_file:

    st.image(
        female_file,
        width=250,
        caption="Female source"
    )

if target_file:

    st.video(
        target_file
    )


# ============================================================
# GENERATE
# ============================================================

generate = st.button(
    "🚀 Generate Synthetic Video",
    type="primary",
    use_container_width=True
)


if generate:

    if male_file is None:

        st.error(
            "Upload the male source image."
        )

        st.stop()

    if female_file is None:

        st.error(
            "Upload the female source image."
        )

        st.stop()

    if target_file is None:

        st.error(
            "Upload the target video."
        )

        st.stop()

    progress = st.progress(0)

    status = st.empty()

    try:

        status.info(
            "Loading models..."
        )

        detector = FaceDetector()

        classifier = GenderClassifier(
            MODEL_PATH
        )

        def update_progress(value):

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
                f"Processing: "
                f"{value * 100:.1f}%"
            )

        output = create_synthetic_video(
            male_file,
            female_file,
            target_file,
            detector,
            classifier,
            update_progress
        )

        progress.progress(1.0)

        status.success(
            "✅ Processing completed!"
        )

        st.header(
            "🎬 Output"
        )

        st.video(
            output
        )

        st.download_button(
            "⬇️ Download Synthetic Video",
            data=output,
            file_name="synthetic_video.mp4",
            mime="video/mp4",
            use_container_width=True
        )

    except Exception as error:

        status.error(
            "❌ Processing failed"
        )

        st.exception(error)


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI-GENERATED / SYNTHETIC MEDIA"
)
