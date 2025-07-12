import torch
from torch import nn
import clip
from PIL import Image
from torchvision.transforms.functional import to_pil_image

from utils import *

class CLIPLoss(nn.Module):
    """
    Этот класс определяет пользовательскую функцию потерь на основе CLIP (Contrastive Language–Image Pre-training).
    Он измеряет сходство между изображением и текстовым описанием.
    """

    def __init__(self, stylegan_size=1024):
        """
        Инициализирует класс CLIPLoss.

        Args:
            opts: Объект, содержащий различные параметры, включая размер изображения StyleGAN.
        """
        super(CLIPLoss, self).__init__()
        # Загружаем предварительно обученную модель CLIP и функцию предварительной обработки
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")

        self.upsample = nn.Upsample(scale_factor=7)
        self.avg_pool = nn.AvgPool2d(kernel_size=stylegan_size // 32)

    def forward(self, image, source_text, target_text):
        """
        Вычисляет потери CLIP между изображением и текстом.

        Args:
            image: Входной тензор изображения.
            source_text: Тензор текстового описания изначальной картинки.
            target_text: Тензор текстового описания картинки, которую мы хотим получить

        Returns:
            Значение потерь CLIP.
        """
        #image = self.preprocess(to_pil_image(image[0])).to(device)
        # Меняем размерность изображения для получения нужного разрешения для CLIP
        image = self.avg_pool(self.upsample(image))
 
        
        source_tokens = clip.tokenize(source_text).to(device)
        target_tokens = clip.tokenize(target_text).to(device)
        
        image_features = self.model.encode_image(image)
        source_text_features = self.model.encode_text(source_tokens)
        target_text_features = self.model.encode_text(target_tokens)
         # Вычислите косинусное расстояние d между эмбеддингами CLIP от картинки и текста и посчитаем loss = 1 - d
        #image_features /image_features.norm(dim=-1, keepdim=True)
        
        loss = 1 - torch.cosine_similarity(
        image_features - source_text_features,
        target_text_features - source_text_features,
        dim=-1
        ).mean()
        return loss
    
import torch
from torch import nn


class IDLoss(nn.Module):
    def __init__(self, model_weights, device='cuda'):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        # Загружаем предобученную модель ResNet ArcFace для извлечения признаков лица.
        # input_size: размер входного изображения (112x112).
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        # Загружаем веса предобученной модели
        self.facenet.load_state_dict(torch.load(model_weights))
        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        # Перемещаем модель ArcFace на GPU, если доступен.
        self.facenet.to(device)


    def extract_feats(self, x):
        # Функция для извлечения эмбеддингов лица из изображения x.
        # Если размер изображения не 256x256, изменяем его с помощью пулинга.
        if x.shape[2] != 256:
            x = self.pool(x)
        # Обрезаем центральную область изображения (35:223, 32:220), содержащую лицо.
        x = x[:, :, 35:223, 32:220]
        # Изменяем размер обрезанной области до 112x112 с помощью пулинга.
        x = self.face_pool(x)
        # Извлекаем эмбеддинги лица с помощью модели ArcFace.
        x_feats = self.facenet(x)
        # Возвращаем извлеченные эмбеддинги.
        return x_feats

    def forward(self, y_hat, y):
        # Функция для вычисления ID Loss.
        # y_hat: отредактированное изображение.
        # y: исходное изображение.
        n_samples = y.shape[0]
        # Извлекаем признаки лица из исходного изображения.
        initial_embeddings = self.extract_feats(y).detach().to(device)

        # Извлекаем признаки лица из отредактированного изображения.
        red_embeddings = self.extract_feats(y_hat).detach().to(device)
        

        # Считаем наш лосс
        loss = 1 - torch.nn.functional.cosine_similarity(initial_embeddings, red_embeddings).detach().to(device)

        return loss