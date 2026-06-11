# Deepfake Detection Pipeline — Technical Breakdown

A detailed walkthrough of how each stage of the Django-based deepfake detection system works, from video upload through to final prediction.

---

## Step 1 — Video Upload & Validation

### Form Definition

```python
# ml_app/forms.py
class VideoUploadForm(forms.Form):
    upload_video_file = forms.FileField(
        label="Select Video",
        required=True,
        widget=forms.FileInput(attrs={"accept": "video/*"})
    )
```

- `accept="video/*"` — browser restricts the file picker to video files
- `FileField` — Django handles file object creation
- `required=True` — file must be present

### Validation Logic (`views.py` — `index` function)

```python
# Configuration in settings.py
CONTENT_TYPES = ['video']
MAX_UPLOAD_SIZE = "104857600"           # 100 MB in bytes
ALLOWED_VIDEO_EXTENSIONS = set(['mp4','gif','webm','avi','3gp','wmv','flv','mkv'])

# POST handler
video_upload_form = VideoUploadForm(request.POST, request.FILES)

if video_upload_form.is_valid():
    video_file = video_upload_form.cleaned_data['upload_video_file']
    video_file_ext = video_file.name.split('.')[-1]
    video_content_type = video_file.content_type.split('/')[0]  # e.g. 'video' from 'video/mp4'

    # Check 1: Content type must be 'video'
    if video_content_type in settings.CONTENT_TYPES:

        # Check 2: File size must be ≤ 100 MB
        if video_file.size > int(settings.MAX_UPLOAD_SIZE):
            video_upload_form.add_error("upload_video_file", "Maximum file size 100 MB")
            return render(request, index_template_name, {"form": video_upload_form})

    # Check 3: Extension must be in ALLOWED_VIDEO_EXTENSIONS
    if allowed_video_file(video_file.name) == False:
        video_upload_form.add_error("upload_video_file", "Only video files are allowed")
        return render(request, index_template_name, {"form": video_upload_form})
```

### Saving the File

```python
# Timestamped filename to prevent overwrites
saved_video_file = 'uploaded_file_' + str(int(time.time())) + "." + video_file_ext
# e.g. uploaded_file_1686754321.mp4

with open(os.path.join(settings.PROJECT_DIR, 'uploaded_videos', saved_video_file), 'wb') as vFile:
    shutil.copyfileobj(video_file, vFile)   # Binary copy to disk

request.session['file_name'] = os.path.join(
    settings.PROJECT_DIR,
    'uploaded_videos',
    saved_video_file
)
```

---

## Step 2 — Frame Extraction & Preprocessing

### Frame Extraction (`predict_page` function)

```python
cap = cv2.VideoCapture(video_file)
frames = []

while cap.isOpened():
    ret, frame = cap.read()     # ret = success bool, frame = BGR numpy array
    if ret:
        frames.append(frame)
    else:
        break

cap.release()
print(f"Number of frames: {len(frames)}")
```

Every frame is read sequentially into memory. Each frame is a `(height, width, 3)` numpy array in BGR format.

### Face Detection & Cropping

```python
FIXED_SEQUENCE_LENGTH = 60      # Only the first 60 frames are processed

for i in range(FIXED_SEQUENCE_LENGTH):
    if i >= len(frames):
        break

    frame = frames[i]

    # 1. Convert BGR → RGB (required by PIL and face_recognition)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 2. Save the preprocessed frame
    image_name = f"{video_file_name_only}_preprocessed_{i+1}.png"
    image_path = os.path.join(settings.PROJECT_DIR, 'uploaded_images', image_name)
    img_rgb = pImage.fromarray(rgb_frame, 'RGB')
    img_rgb.save(image_path)
    preprocessed_images.append(image_name)

    # 3. Detect face using HOG model
    face_locations = face_recognition.face_locations(rgb_frame, model='hog')
    # Returns: [(top, right, bottom, left), ...]

    if len(face_locations) == 0:
        continue    # Skip frames with no detected face

    # 4. Crop face with 30% margin
    rgb_face = crop_face_with_margin(rgb_frame, face_locations[0])

    # 5. Save cropped face
    img_face_rgb = pImage.fromarray(rgb_face, 'RGB')
    image_name = f"{video_file_name_only}_cropped_faces_{i+1}.png"
    image_path = os.path.join(settings.PROJECT_DIR, 'uploaded_images', image_name)
    img_face_rgb.save(image_path)
    faces_found += 1
    faces_cropped_images.append(image_name)
```

### `crop_face_with_margin` helper

```python
def crop_face_with_margin(frame, face_location, margin=0.3):
    top, right, bottom, left = face_location
    height = bottom - top
    width  = right - left

    y1 = max(0, int(top    - height * 0.3))
    x1 = max(0, int(left   - width  * 0.3))
    y2 = min(frame.shape[0], int(bottom + height * 0.3))
    x2 = min(frame.shape[1], int(right  + width  * 0.3))

    return frame[y1:y2, x1:x2]
```

**Output:** for a 500-frame video, 60 frames are processed. Each frame and each detected face is saved as a PNG in `uploaded_images/`.

---

## Step 3 — Tensor Creation & Normalization

### `validation_dataset` class

```python
class validation_dataset(Dataset):
    def __init__(self, video_names, sequence_length=60, transform=None):
        self.video_names   = video_names
        self.transform     = transform
        self.count         = sequence_length

    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []
        faces_detected = 0

        for i, frame in enumerate(self.frame_extract(video_path)):
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_recognition.face_locations(rgb_frame, model='hog')

            if faces:
                frame = crop_face_with_margin(rgb_frame, faces[0])
                faces_detected += 1
            else:
                frame = rgb_frame   # Fallback to full frame

            frames.append(self.transform(frame))

            if len(frames) == self.count:
                break

        # Zero-pad if fewer than 60 frames were found
        while len(frames) < self.count:
            frames.append(torch.zeros_like(frames[0]))

        frames = torch.stack(frames)            # → [60, 3, 112, 112]
        frames = frames[:self.count]
        return frames.unsqueeze(0)              # → [1, 60, 3, 112, 112]
```

### Transform Pipeline

```python
im_size = 112
mean = [0.485, 0.456, 0.406]    # ImageNet channel means
std  = [0.229, 0.224, 0.225]    # ImageNet channel stds

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),  # Resize to 112 × 112
    transforms.ToTensor(),                  # Scale to [0, 1]
    transforms.Normalize(mean, std)         # Standardize per channel
])
```

**Normalization example:**

```
Original pixel:   [255, 128, 64]

After ToTensor(): [1.0,  0.502, 0.251]

After Normalize():
  Channel R: (1.000 - 0.485) / 0.229 =  2.25
  Channel G: (0.502 - 0.456) / 0.224 =  0.21
  Channel B: (0.251 - 0.406) / 0.225 = -0.69

Result: [2.25, 0.21, -0.69]
```

**Final tensor shape:** `[1, 60, 3, 112, 112]` — 1 batch × 60 frames × 3 channels × 112 × 112 pixels.

---

## Step 4 — Primary Model Prediction

### Architecture

```python
class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1,
                 hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()

        # CNN feature extractor (ResNeXt-50, pretrained on ImageNet)
        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        # Removes final avg-pool + classifier; keeps conv layers → 2048-dim feature maps

        # Temporal sequence modelling
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional=bidirectional)

        # Classification head
        self.relu    = nn.LeakyReLU()
        self.dp      = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)     # 2048 → 2 (REAL / FAKE)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape      # [1, 60, 3, 112, 112]

        x = x.view(batch_size * seq_length, c, h, w)   # [60, 3, 112, 112]

        fmap = self.model(x)                            # → [60, 2048, 7, 7]
        x    = self.avgpool(fmap)                       # → [60, 2048, 1, 1]
        x    = x.view(batch_size, seq_length, 2048)     # → [1, 60, 2048]

        x_lstm, _ = self.lstm(x, None)                  # → [1, 60, 2048]

        # Return feature maps + classification logits from last LSTM step
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))
```

### Running Inference

```python
def predict(model, img, path='./', video_file_name=""):
    model.eval()

    with torch.no_grad():
        fmap, logits = model(img.to(device))
        # logits: [1, 2], e.g. [[5.2, -3.1]]

        probabilities = F.softmax(logits, dim=1)
        # e.g. [[0.99, 0.01]]

        confidence, prediction = torch.max(probabilities, dim=1)
        # confidence = 0.99,  prediction = 1 (REAL)

    return [int(prediction.item()), confidence.item() * 80]
    # Confidence is scaled by 80 (not 100) — a model-specific calibration factor
    # e.g. 0.99 × 80 = 79.2%
```

---

## Step 5 — GenConViT Fallback (Optional)

```python
if USE_GENCONVIT and primary_prediction[1] < GENCONVIT_FALLBACK_CONFIDENCE:
    try:
        prediction = predict_genconvit_video(path_to_videos[i], sequence_length)
    except Exception as genconvit_error:
        prediction = primary_prediction     # Fall back to primary result on error
```

### GenConViT Config (`config.yaml`)

```yaml
model:
  backbone: convnext_tiny
  embedder: swin_tiny_patch4_window7_224
  latent_dims: 12544
```

```python
def predict_genconvit_video(video_file, sequence_length=60):
    model = load_genconvit_model()

    faces = df_face(video_file, num_frames)
    if len(faces) < 1:
        raise RuntimeError("GenConViT could not detect a face")

    prediction_index, score = pred_vid(faces, model)
    label      = real_or_fake(prediction_index)
    confidence = max(0.0, min(float(score) * 100, 100.0))

    return [1 if label == "REAL" else 0, confidence]
```

---

## Step 6 — Result Display

```python
confidence = round(prediction[1], 1)
output     = class_labels.get(prediction[0], "UNKNOWN")    # 0 → 'FAKE', 1 → 'REAL'

context = {
    'preprocessed_images':  preprocessed_images,
    'faces_cropped_images': faces_cropped_images,
    'heatmap_images':       heatmap_images,
    'original_video':       production_video_name,
    'output':               output,       # "REAL" or "FAKE"
    'confidence':           confidence,   # e.g. 79.2
}

return render(request, predict_template_name, context)
```

The template renders: the original video player, a grid of 60 preprocessed frames, all detected face crops, and the final verdict — e.g. **FAKE — 79.2% confidence**.

---

## Complete Pipeline

```
USER SUBMITS VIDEO
        │
        ▼
VideoUploadForm.is_valid()
  ├─ FileField present?
  └─ (Django built-in validation)
        │
        ▼
allowed_video_file(filename)
  └─ ext in ALLOWED_VIDEO_EXTENSIONS?
        │
        ▼
video_file.size > 104857600 (100 MB)?
  └─ Yes → add error, re-render form
        │
        ▼
video_content_type == 'video'?
  └─ No  → add error, re-render form
        │
        ▼
SAVE  shutil.copyfileobj()
      → /uploaded_videos/uploaded_file_{timestamp}.{ext}
        │
        ▼
STORE request.session['file_name']
        │
        ▼
REDIRECT → predict_page
        │
        ▼
cv2.VideoCapture(video_path).read()
  └─ Extract ALL frames into memory
        │
        ▼
FOR i IN range(60):
  ├─ cv2.cvtColor(BGR → RGB)
  ├─ SAVE  PIL Image → uploaded_images/{name}_preprocessed_{i}.png
  ├─ face_recognition.face_locations(model='hog')
  └─ IF face found:
      ├─ crop_face_with_margin(margin=0.3)
      └─ SAVE  PIL Image → uploaded_images/{name}_cropped_faces_{i}.png
        │
        ▼
faces_found == 0?
  └─ Yes → RENDER "No faces detected"
        │
        ▼
Load models/model.pt  (ResNeXt-50 + LSTM)
        │
        ▼
FOR each frame:
  ├─ transforms.Resize(112, 112)
  ├─ transforms.Normalize(ImageNet mean/std)
  └─ torch.stack() → [1, 60, 3, 112, 112]
        │
        ▼
Model.forward([1, 60, 3, 112, 112])
  ├─ ResNeXt-50 → [60, 2048, 7, 7]
  ├─ AvgPool    → [60, 2048]
  ├─ LSTM       → [1,  60, 2048]
  └─ Linear     → [1,  2]   (logits)
        │
        ▼
Softmax(logits) → probabilities
        │
        ▼
torch.max() → prediction index + confidence
        │
        ▼
confidence < threshold AND USE_GENCONVIT?
  └─ Yes → predict_genconvit_video()
        │
        ▼
RETURN [0/1, confidence_%]
        │
        ▼
RENDER predict.html
  ├─ Original video player
  ├─ 60 preprocessed frames
  ├─ Detected face crops
  ├─ Verdict:     "FAKE" or "REAL"
  └─ Confidence:  X%
```

---

## Key Implementation Notes

1. **Validation** occurs at four levels: Django form, content-type check, file-size check, and extension check.
2. **Frame extraction** reads all frames but only processes the first 60.
3. **Normalization** uses ImageNet statistics so input distribution matches ResNeXt-50's training data.
4. **LSTM** processes the temporal sequence of 60 frames to detect inconsistencies that appear across time — a key signal in deepfakes.
5. **Confidence scaling** multiplies the softmax output by 80, not 100 — a calibration choice specific to this model.
6. **GenConViT fallback** provides a second opinion from a ConvNeXt + Swin Transformer ensemble when the primary model's confidence is low.
