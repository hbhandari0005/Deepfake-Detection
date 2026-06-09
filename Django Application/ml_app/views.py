from django.shortcuts import render, redirect
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force CPU for Keras to avoid GPU conflicts

print("[INFO] Keras forced to CPU-only mode to avoid GPU memory conflicts")

import torch
# Force PyTorch to use CPU to avoid GPU conflicts with Keras
torch.cuda.is_available = lambda: False
device = torch.device('cpu')

print("[INFO] PyTorch forced to CPU-only mode")

import torchvision
import torchvision.utils as vutils
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import numpy as np
import cv2
import matplotlib.pyplot as plt
import face_recognition
from torch.autograd import Variable
import time
import sys
from torch import nn
import torch.nn.functional as F
import copy
from torchvision import models
import shutil
from PIL import Image as pImage
from django.conf import settings
from .forms import VideoUploadForm

FIXED_SEQUENCE_LENGTH = 60
USE_GENCONVIT = True
GENCONVIT_NET = "genconvit"
GENCONVIT_MODEL = None

index_template_name = 'index.html'
predict_template_name = 'predict.html'
about_template_name = "about.html"

im_size = 112
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]
inv_normalize =  transforms.Normalize(mean=-1*np.divide(mean,std),std=np.divide([1,1,1],std))

class_labels = {0: 'FAKE', 1: 'REAL'}
face_crop_margin = 0.3

train_transforms = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize((im_size,im_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean,std)])

def load_genconvit_model():
    global GENCONVIT_MODEL

    genconvit_src = os.path.join(settings.PROJECT_DIR, "genconvit_src")
    ed_weight = os.path.join(settings.PROJECT_DIR, "genconvit_weights", "genconvit_ed_inference.pth")
    vae_weight = os.path.join(settings.PROJECT_DIR, "genconvit_weights", "genconvit_vae_inference.pth")

    if not os.path.isdir(genconvit_src):
        raise FileNotFoundError(f"GenConViT source folder not found: {genconvit_src}")
    if not os.path.exists(ed_weight):
        raise FileNotFoundError(f"GenConViT ED weight not found: {ed_weight}")
    if not os.path.exists(vae_weight):
        raise FileNotFoundError(f"GenConViT VAE weight not found: {vae_weight}")

    if genconvit_src not in sys.path:
        sys.path.insert(0, genconvit_src)

    from model.config import load_config
    from model.pred_func import load_genconvit

    if GENCONVIT_MODEL is None:
        print("[INFO] Loading GenConViT weights")
        GENCONVIT_MODEL = load_genconvit(
            load_config(),
            GENCONVIT_NET,
            ed_weight,
            vae_weight,
            fp16=False,
        )

    return GENCONVIT_MODEL

def predict_genconvit_video(video_file, sequence_length=FIXED_SEQUENCE_LENGTH):
    genconvit_src = os.path.join(settings.PROJECT_DIR, "genconvit_src")
    if genconvit_src not in sys.path:
        sys.path.insert(0, genconvit_src)

    from model.pred_func import df_face, pred_vid, real_or_fake

    num_frames = int(sequence_length)
    model = load_genconvit_model()
    faces = df_face(video_file, num_frames)
    if len(faces) < 1:
        raise RuntimeError("GenConViT could not detect a face in the selected video frames")

    prediction_index, score = pred_vid(faces, model)
    label = real_or_fake(prediction_index)
    confidence = max(0.0, min(float(score) * 100, 100.0))
    print(f"[INFO] GenConViT prediction: {label} ({confidence:.1f}%)")
    return [1 if label == "REAL" else 0, confidence]

def crop_face_with_margin(frame, face_location, margin=face_crop_margin):
    top, right, bottom, left = face_location
    height = bottom - top
    width = right - left
    y1 = max(0, int(top - height * margin))
    x1 = max(0, int(left - width * margin))
    y2 = min(frame.shape[0], int(bottom + height * margin))
    x2 = min(frame.shape[1], int(right + width * margin))
    return frame[y1:y2, x1:x2]

def save_debug_face_samples(frames_tensor, video_file_name=""):
    if frames_tensor.ndim != 5:
        raise ValueError(f"Expected input tensor shape [batch, seq, C, H, W], got {tuple(frames_tensor.shape)}")
    if frames_tensor.shape[0] != 1 or frames_tensor.shape[2:] != (3, im_size, im_size):
        raise ValueError(
            f"Expected input tensor shape [1, seq, 3, {im_size}, {im_size}], "
            f"got {tuple(frames_tensor.shape)}"
        )

    print(f"[DEBUG] Input tensor shape: {frames_tensor.shape}")
    print(
        f"[DEBUG] Tensor min: {frames_tensor.min().item():.3f}, "
        f"max: {frames_tensor.max().item():.3f}, "
        f"mean: {frames_tensor.mean().item():.3f}"
    )

    sample_frames = frames_tensor[0, :5]
    debug_dir = os.path.join(settings.PROJECT_DIR, 'uploaded_images')
    os.makedirs(debug_dir, exist_ok=True)
    mean_tensor = torch.tensor(mean).view(3, 1, 1)
    std_tensor = torch.tensor(std).view(3, 1, 1)

    for i, frame in enumerate(sample_frames):
        denorm = (frame.cpu() * std_tensor + mean_tensor).clamp(0, 1)
        image_name = f"{video_file_name}_debug_face_{i}.png" if video_file_name else f"debug_face_{i}.png"
        vutils.save_image(denorm, os.path.join(debug_dir, image_name))

    print("[DEBUG] Saved 5 face crop samples to uploaded_images/*debug_face_0..4.png")

class Model(nn.Module):

    def __init__(self, num_classes,latent_dim= 2048, lstm_layers=1 , hidden_dim = 2048, bidirectional = False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained = True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim,hidden_dim, lstm_layers,  bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048,num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size,seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size,seq_length,2048)
        x_lstm,_ = self.lstm(x,None)
        return fmap,self.dp(self.linear1(x_lstm[:,-1,:]))


class validation_dataset(Dataset):
    def __init__(self,video_names,sequence_length=60,transform = None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self,idx):
        video_path = self.video_names[idx]
        frames = []
        faces_detected = 0
        for i,frame in enumerate(self.frame_extract(video_path)):
            #if(i % a == first_frame):
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = face_recognition.face_locations(rgb_frame, model='hog')
            if faces:
              frame = crop_face_with_margin(rgb_frame, faces[0])
              faces_detected += 1
            else:
              frame = rgb_frame
              print(f"[WARNING] No face detected in frame {i}, using full frame as fallback")
            frames.append(self.transform(frame))
            if(len(frames) == self.count):
                break
        """
        for i,frame in enumerate(self.frame_extract(video_path)):
            if(i % a == first_frame):
                frames.append(self.transform(frame))
        """        
        if not frames:
            raise ValueError("No frames could be extracted from the uploaded video.")
        while len(frames) < self.count:
            frames.append(torch.zeros_like(frames[0]))
        print(f"[DEBUG] Frames analyzed: {len(frames[:self.count])}, faces detected: {faces_detected}")
        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)
    
    def frame_extract(self,path):
      vidObj = cv2.VideoCapture(path) 
      success = 1
      while success:
          success, image = vidObj.read()
          if success:
              yield image

def im_convert(tensor, video_file_name):
    """ Display a tensor as an image. """
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image = image.numpy()
    image = image.transpose(1,2,0)
    image = image.clip(0, 1)
    # This image is not used
    # cv2.imwrite(os.path.join(settings.PROJECT_DIR, 'uploaded_images', video_file_name+'_convert_2.png'),image*255)
    return image

def im_plot(tensor):
    image = tensor.cpu().numpy().transpose(1,2,0)
    b,g,r = cv2.split(image)
    image = cv2.merge((r,g,b))
    image = image*[0.22803, 0.22145, 0.216989] +  [0.43216, 0.394666, 0.37645]
    image = image*255.0
    plt.imshow(image.astype('uint8'))
    plt.show()


def predict(model,img,path = './', video_file_name=""):
  model.eval()
  save_debug_face_samples(img, video_file_name)
  with torch.no_grad():
    fmap, logits = model(img.to(device))
    probabilities = F.softmax(logits, dim=1)
    confidence, prediction = torch.max(probabilities, dim=1)
  img = im_convert(img[:,-1,:,:,:], video_file_name)
  print(f"[DEBUG] Raw model output: {logits.detach().cpu()}")
  print(f"[DEBUG] After softmax: {probabilities.detach().cpu()}")
#   print(f"[DEBUG] Predicted class index: {prediction.item()}")
#   print(f"[DEBUG] Predicted label: {class_labels.get(prediction.item(), 'UNKNOWN')}")
  print('confidence of prediction:', confidence.item()*100)
  return [int(prediction.item()), confidence.item()*100]


def plot_heat_map(i, model, img, path = './', video_file_name=''):
  fmap,logits = model(img.to(device))
  params = list(model.parameters())
  weight_softmax = model.linear1.weight.detach().cpu().numpy()
  logits = F.softmax(logits, dim=1)
  _,prediction = torch.max(logits,1)
  idx = np.argmax(logits.detach().cpu().numpy())
  bz, nc, h, w = fmap.shape
  #out = np.dot(fmap[-1].detach().cpu().numpy().reshape((nc, h*w)).T,weight_softmax[idx,:].T)
  out = np.dot(fmap[i].detach().cpu().numpy().reshape((nc, h*w)).T,weight_softmax[idx,:].T)
  predict = out.reshape(h,w)
  predict = predict - np.min(predict)
  predict_img = predict / np.max(predict)
  predict_img = np.uint8(255*predict_img)
  out = cv2.resize(predict_img, (im_size,im_size))
  heatmap = cv2.applyColorMap(out, cv2.COLORMAP_JET)
  img = im_convert(img[:,-1,:,:,:], video_file_name)
  result = heatmap * 0.5 + img*0.8*255
  # Saving heatmap - Start
  heatmap_name = video_file_name+"_heatmap_"+str(i)+".png"
  image_name = os.path.join(settings.PROJECT_DIR, 'uploaded_images', heatmap_name)
  cv2.imwrite(image_name,result)
  # Saving heatmap - End
  result1 = heatmap * 0.5/255 + img*0.8
  r,g,b = cv2.split(result1)
  result1 = cv2.merge((r,g,b))
  return image_name

ALLOWED_VIDEO_EXTENSIONS = set(['mp4','gif','webm','avi','3gp','wmv','flv','mkv'])

def allowed_video_file(filename):
    #print("filename" ,filename.rsplit('.',1)[1].lower())
    if (filename.rsplit('.',1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS):
        return True
    else: 
        return False
def index(request):
    if request.method == 'GET':
        video_upload_form = VideoUploadForm()
        if 'file_name' in request.session:
            del request.session['file_name']
        if 'preprocessed_images' in request.session:
            del request.session['preprocessed_images']
        if 'faces_cropped_images' in request.session:
            del request.session['faces_cropped_images']
        return render(request, index_template_name, {"form": video_upload_form})
    else:
        video_upload_form = VideoUploadForm(request.POST, request.FILES)
        if video_upload_form.is_valid():
            video_file = video_upload_form.cleaned_data['upload_video_file']
            video_file_ext = video_file.name.split('.')[-1]
            sequence_length = FIXED_SEQUENCE_LENGTH
            video_content_type = video_file.content_type.split('/')[0]
            if video_content_type in settings.CONTENT_TYPES:
                if video_file.size > int(settings.MAX_UPLOAD_SIZE):
                    video_upload_form.add_error("upload_video_file", "Maximum file size 100 MB")
                    return render(request, index_template_name, {"form": video_upload_form})
            
            if allowed_video_file(video_file.name) == False:
                video_upload_form.add_error("upload_video_file","Only video files are allowed ")
                return render(request, index_template_name, {"form": video_upload_form})
            
            saved_video_file = 'uploaded_file_'+str(int(time.time()))+"."+video_file_ext
            if settings.DEBUG:
                with open(os.path.join(settings.PROJECT_DIR, 'uploaded_videos', saved_video_file), 'wb') as vFile:
                    shutil.copyfileobj(video_file, vFile)
                request.session['file_name'] = os.path.join(settings.PROJECT_DIR, 'uploaded_videos', saved_video_file)
            else:
                with open(os.path.join(settings.PROJECT_DIR, 'uploaded_videos','app','uploaded_videos', saved_video_file), 'wb') as vFile:
                    shutil.copyfileobj(video_file, vFile)
                request.session['file_name'] = os.path.join(settings.PROJECT_DIR, 'uploaded_videos','app','uploaded_videos', saved_video_file)
            request.session['sequence_length'] = sequence_length
            return redirect('ml_app:predict')
        else:
            return render(request, index_template_name, {"form": video_upload_form})

def predict_page(request):
    if request.method == "GET":
        # Redirect to 'home' if 'file_name' is not in session
        if 'file_name' not in request.session:
            return redirect("ml_app:home")
        if 'file_name' in request.session:
            video_file = request.session['file_name']
        sequence_length = FIXED_SEQUENCE_LENGTH
        path_to_videos = [video_file]
        video_file_name = os.path.basename(video_file)
        video_file_name_only = os.path.splitext(video_file_name)[0]
        # Production environment adjustments
        if not settings.DEBUG:
            production_video_name = os.path.join('/home/app/staticfiles/', video_file_name.split('/')[3])
            print("Production file name", production_video_name)
        else:
            production_video_name = video_file_name

        video_dataset = None
        model = None
        start_time = time.time()
        # Display preprocessing images
        print("<=== | Started Videos Splitting | ===>")
        preprocessed_images = []
        faces_cropped_images = []
        cap = cv2.VideoCapture(video_file)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                break
        cap.release()

        print(f"Number of frames: {len(frames)}")
        # Process each frame for preprocessing and face cropping
        faces_found = 0
        for i in range(sequence_length):
            if i >= len(frames):
                break
            frame = frames[i]

            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Save preprocessed image
            image_name = f"{video_file_name_only}_preprocessed_{i+1}.png"
            image_path = os.path.join(settings.PROJECT_DIR, 'uploaded_images', image_name)
            img_rgb = pImage.fromarray(rgb_frame, 'RGB')
            img_rgb.save(image_path)
            preprocessed_images.append(image_name)

            # Face detection and cropping
            face_locations = face_recognition.face_locations(rgb_frame, model='hog')
            if len(face_locations) == 0:
                continue

            rgb_face = crop_face_with_margin(rgb_frame, face_locations[0])
            img_face_rgb = pImage.fromarray(rgb_face, 'RGB')
            image_name = f"{video_file_name_only}_cropped_faces_{i+1}.png"
            image_path = os.path.join(settings.PROJECT_DIR, 'uploaded_images', image_name)
            img_face_rgb.save(image_path)
            faces_found += 1
            faces_cropped_images.append(image_name)

        print("<=== | Videos Splitting and Face Cropping Done | ===>")
        print("--- %s seconds ---" % (time.time() - start_time))

        # No face detected
        if faces_found == 0:
            return render(request, predict_template_name, {"no_faces": True})

        # Perform prediction
        try:
            heatmap_images = []
            output = ""
            confidence = 0.0

            for i in range(len(path_to_videos)):
                print("<=== | Started Prediction | ===>")
                prediction = predict_genconvit_video(path_to_videos[i], sequence_length)
                confidence = round(prediction[1], 1)
                output = class_labels.get(prediction[0], "UNKNOWN")
                # print("Prediction:", prediction[0], "==", output, "Confidence:", confidence)
                print("Confidence:", confidence)
                print("<=== | Prediction Done | ===>")
                print("--- %s seconds ---" % (time.time() - start_time))

                # Uncomment if you want to create heat map images
                # for j in range(sequence_length):
                #     heatmap_images.append(plot_heat_map(j, model, video_dataset[i], './', video_file_name_only))

            # Cleanup memory after prediction to free up GPU/CPU resources
            import gc
            if 'model' in locals() and model is not None:
                del model
            if 'video_dataset' in locals():
                del video_dataset
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Render results
            context = {
                'preprocessed_images': preprocessed_images,
                'faces_cropped_images': faces_cropped_images,
                'heatmap_images': heatmap_images,
                'original_video': production_video_name,
                'models_location': os.path.join(settings.PROJECT_DIR, 'models'),
                'output': output,
                'confidence': confidence
            }

            if settings.DEBUG:
                return render(request, predict_template_name, context)
            else:
                return render(request, predict_template_name, context)

        except Exception as e:
            print(f"Exception occurred during prediction: {e}")
            return render(request, 'cuda_full.html')
def about(request):
    return render(request, about_template_name)

def handler404(request,exception):
    return render(request, '404.html', status=404)
def cuda_full(request):
    return render(request, 'cuda_full.html')
