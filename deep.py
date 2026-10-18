import streamlit as st
from pathlib import Path
import tempfile
import subprocess
import cv2
import numpy as np
import os


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="AI Synthetic Media Generator",
    page_icon="🎭",
    layout="wide"
)


# ============================================================
# BACKEND
# ============================================================

WATERMARK_TEXT = "AI-GENERATED / SYNTHETIC MEDIA"

GENDER_CONFIDENCE_THRESHOLD = 0.70

# gender_net.onnx expected:
# 0 = female
# 1 = male
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
                "OpenCV Haar Cascade could not be loaded."
            )

    def detect(self, image):

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

        if len(faces) == 0:

            raise ValueError(
                "No face detected in source image."
            )

        return max(
            faces,
            key=lambda box:
            (box[2] - box[0])
            *
            (box[3] - box[1])
        )


# ============================================================
# GENDER CLASSIFIER
# ============================================================

class GenderClassifier:

    def __init__(self, model_path):

        self.net = cv2.dnn.readNetFromONNX(
            str(model_path)
        )

    def crop_face(
        self,
        image,
        bbox
    ):

        x1, y1, x2, y2 = bbox

        h, w = image.shape[:2]

        x1 = max(0, x1)
        y1 = max(0, y1)

        x2 = min(w, x2)
        y2 = min(h, y2)

        if x2 <= x1 or y2 <= y1:

            return None

        face = image[
            y1:y2,
            x1:x2
        ]

        if face.size == 0:

            return None

        return face

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

        blob = cv2.dnn.blobFromImage(
            face,
            scalefactor=1.0,
            size=(224, 224),
            mean=(104, 117, 123),
            swapRB=False,
            crop=False
        )

        self.net.setInput(blob)

        output = self.net.forward()

        scores = np.asarray(
            output
        ).flatten()

        if len(scores) < 2:

            return "unknown", 0.0

        female_score = float(
            scores[GENDER_INDEX["female"]]
        )

        male_score = float(
            scores[GENDER_INDEX["male"]]
        )

        values = np.array(
            [
                female_score,
                male_score
            ],
            dtype=np.float32
        )

        # Convert logits to probabilities
        if (
            np.any(values < 0.0)
            or np.any(values > 1.0)
            or abs(
                float(values.sum()) - 1.0
            ) > 0.10
        ):

            values = np.exp(
                values - np.max(values)
            )

            values = (
                values /
                np.sum(values)
            )

        female_score = float(values[0])
        male_score = float(values[1])

        if male_score > female_score:

            gender = "male"
            confidence = male_score

        else:

            gender = "female"
            confidence = female_score

        if confidence < GENDER_CONFIDENCE_THRESHOLD:

            return "unknown", confidence

        return gender, confidence


# ============================================================
# SOURCE FACE EXTRACTION
# ============================================================

def extract_source_face(
    image,
    bbox
):

    x1, y1, x2, y2 = bbox

    face = image[
        y1:y2,
        x1:x2
    ]

    if face.size == 0:

        raise ValueError(
            "Source face could not be extracted."
        )

    return face


# ============================================================
# FACE REGION REPLACEMENT
# ============================================================

def replace_face(
    frame,
    target_bbox,
    source_face
):

    x1, y1, x2, y2 = target_bbox

    width = x2 - x1
    height = y2 - y1

    if width <= 0 or height <= 0:

        return frame

    # Resize source face
    source = cv2.resize(
        source_face,
        (width, height),
        interpolation=cv2.INTER_AREA
    )

    target = frame[
        y1:y2,
        x1:x2
    ]

    if target.size == 0:

        return frame

    # Create elliptical face mask
    mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    center = (
        width // 2,
        height // 2
    )

    axes = (
        max(1, int(width * 0.42)),
        max(1, int(height * 0.47))
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

    # Smooth mask
    blur_size = 21

    if width < 40 or height < 40:

        blur_size = 9

    mask = cv2.GaussianBlur(
        mask,
        (
            blur_size,
            blur_size
        ),
        0
    )

    alpha = (
        mask.astype(np.float32)
        / 255.0
    )

    alpha = alpha[:, :, None]

    blended = (
        source.astype(np.float32)
        * alpha
        +
        target.astype(np.float32)
        * (1.0 - alpha)
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
        WATERMARK_TEXT,
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
    detector,
    gender_classifier,
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

        fps = 25

    if width <= 0 or height <= 0:

        cap.release()

        raise RuntimeError(
            "Invalid video dimensions."
        )

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
            "Could not create temporary video."
        )

    frame_number = 0

    try:

        while True:

            success, frame = cap.read()

            if not success:

                break

            # Detect faces
            faces = detector.detect(
                frame
            )

            result = frame.copy()

            # Process each face
            for bbox in faces:

                gender, confidence = (
                    gender_classifier.predict(
                        frame,
                        bbox
                    )
                )

                # Male → Male
                if gender == "male":

                    result = replace_face(
                        result,
                        bbox,
                        male_source_face
                    )

                # Female → Female
                elif gender == "female":

                    result = replace_face(
                        result,
                        bbox,
                        female_source_face
                    )

            # Add disclosure
            result = add_watermark(
                result
            )

            writer.write(
                result
            )

            frame_number += 1

            progress = (
                frame_number /
                max(total_frames, 1)
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
            "FFmpeg error:\n\n"
            + result.stderr[-3000:]
        )

    return final_output


# ============================================================
# MAIN BACKEND FUNCTION
# ============================================================

def create_synthetic_video(
    male_file,
    female_file,
    target_file,
    gender_model_file,
    progress_callback=None
):

    with tempfile.TemporaryDirectory() as temp:

        temp = Path(temp)

        # Temporary input files
        male_path = (
            temp / "male_source.jpg"
        )

        female_path = (
            temp / "female_source.jpg"
        )

        target_path = (
            temp / "target.mp4"
        )

        model_path = (
            temp / "gender_net.onnx"
        )

        # Temporary video files
        processed_path = (
            temp / "processed.mp4"
        )

        final_path = (
            temp / "synthetic_video.mp4"
        )

        # Save uploads temporarily
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

        with open(
            model_path,
            "wb"
        ) as f:

            f.write(
                gender_model_file.getbuffer()
            )

        # Read source images
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

        # Load detector
        detector = FaceDetector()

        # Load gender classifier
        gender_classifier = GenderClassifier(
            model_path
        )

        # Find source faces
        male_bbox = detector.largest(
            male_image
        )

        female_bbox = detector.largest(
            female_image
        )

        # Extract source faces
        male_source_face = extract_source_face(
            male_image,
            male_bbox
        )

        female_source_face = extract_source_face(
            female_image,
            female_bbox
        )

        # Process video
        process_video(

            target_path,

            processed_path,

            detector,

            gender_classifier,

            male_source_face,

            female_source_face,

            progress_callback
        )

        # Restore original audio
        combine_video_audio(

            processed_path,

            target_path,

            final_path
        )

        # Read final video into memory
        with open(
            final_path,
            "rb"
        ) as f:

            result_bytes = f.read()

    return result_bytes


# ============================================================
# STREAMLIT APP
# ============================================================

st.title(
    "🎭 AI Synthetic Media Generator"
)

st.write(
    "Authorized / consenting face-swap demonstration"
)

st.warning(
    "Use only faces and videos you have permission to process. "
    "The generated video contains a synthetic-media watermark."
)


# ============================================================
# UPLOAD SECTION
# ============================================================

st.header("1. Upload Source Images")

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


st.header("2. Upload Target Video")

target_file = st.file_uploader(
    "🎬 Target Video",
    type=[
        "mp4",
        "mov",
        "avi"
    ]
)


st.header("3. Upload Gender Model")

gender_model_file = st.file_uploader(
    "🧠 gender_net.onnx",
    type=[
        "onnx"
    ]
)


# ============================================================
# PREVIEWS
# ============================================================

if male_file:

    st.subheader("Male Source")

    st.image(
        male_file,
        width=300
    )


if female_file:

    st.subheader("Female Source")

    st.image(
        female_file,
        width=300
    )


if target_file:

    st.subheader("Target Video")

    st.video(
        target_file
    )


# ============================================================
# GENERATE
# ============================================================

st.header("4. Generate")

generate_button = st.button(
    "🚀 Generate Synthetic Video",
    type="primary",
    use_container_width=True
)


if generate_button:

    # Validation

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

    if gender_model_file is None:

        st.error(
            "Please upload gender_net.onnx."
        )

        st.stop()

    # Progress

    progress_bar = st.progress(
        0
    )

    status = st.empty()

    try:

        def update_progress(
            progress
        ):

            value = max(
                0.0,
                min(
                    1.0,
                    float(progress)
                )
            )

            progress_bar.progress(
                value
            )

            status.info(
                f"Processing: "
                f"{value * 100:.1f}%"
            )

        # Run backend

        video_bytes = create_synthetic_video(

            male_file,

            female_file,

            target_file,

            gender_model_file,

            update_progress
        )

        progress_bar.progress(
            1.0
        )

        status.success(
            "✅ Video processing completed!"
        )

        # Output

        st.header(
            "🎬 Generated Video"
        )

        st.video(
            video_bytes
        )

        # Download

        st.download_button(

            label="⬇️ Download Synthetic Video",

            data=video_bytes,

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