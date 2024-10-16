import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2

class image_comparator_v1:
    
    def __init__(self):
        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        self.threshold = 25
        # Load the pretrained model
        self.model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x16d_swsl')
        self.model = self.model.to(self.device)
        # Set the model to evaluation mode
        self.model.eval()
        
    def preprocess_image(self, image): #image must be PIL image
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        return transform(image).unsqueeze(0).to(self.device)
        
    def dissimilarity(self, image1, image2): #image1 and image2 must be OpenCV/numpy images
        #Convert OpenCV/numpy images to PIL images
        image1 = Image.fromarray(image1)
        image2 = Image.fromarray(image2)
        
        #Preprocessing the images
        image1 = self.preprocess_image(image1)
        image2 = self.preprocess_image(image2)
        
        # Get the model predictions for the two images
        output1 = None
        output2 = None
        with torch.no_grad():
            output1 = self.model(image1)
            output2 = self.model(image2)

        # Using the outputs as the embeddings 
        embedding1 = output1
        embedding2 = output2

        # Calculate the L2 norm (Euclidean distance) between the two embeddings
        dissimilarity_score = F.pairwise_distance(embedding1, embedding2)
        
        return dissimilarity_score.item()
    
    def is_same(self, image1, image2):
        dis = self.dissimilarity(image1, image2)
        return (dis <= self.threshold, dis)

if __name__ == "__main__":
    image1 = cv2.imread("./media/oc_1.jpg")
    #image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)

    image2 = cv2.imread("./media/oc_2a.jpg")
    #image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    
    ic = image_comparator_v1()
    print(ic.is_same(image1, image2))