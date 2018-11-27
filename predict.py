import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms, models
import numpy as np
from collections import OrderedDict
import argparse,copy,os,random,time,json
from PIL import Image
from public import load_checkpoint, load_cat_to_name 


def process_image(image):

    # TODO: Process a PIL image for use in a PyTorch model  
    size = [0, 0]

    if image.size[0] > image.size[1]:
        size = [image.size[0], 256]
    else:
        size = [256, image.size[1]]
    
    image.thumbnail(size, Image.ANTIALIAS)
    width, height = image.size  

    LEFT = TOP =  (256 - 224)/2
    RIGHT = BOTTOM =  (256 + 224)/2
    image = image.crop((LEFT, TOP, RIGHT, BOTTOM))
    
    image = np.array(image)
    image = image/255.
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    image = np.transpose(image, (2, 0, 1))
    
    return image


def predict(image_path, model, topk, gpu):
    # TODO: Implement the code to predict the class from an image file
    use_gpu = torch.cuda.is_available()
    if gpu and use_gpu:
        model = model.cuda()
    else:
        model = model.cpu()
    model.eval()    
    image = torch.FloatTensor([process_image(Image.open(image_path))])
    output = model.forward(Variable(image))
    pobabilities = torch.exp(output).data.numpy()[0]

    top_idx = np.argsort(pobabilities)[-topk:][::-1] 
    top_class = [idx_to_class[x] for x in top_idx]
    top_probability = pobabilities[top_idx]

    return top_probability, top_class


def todo_args():
    # TODO: Help to run the project
    parser = argparse.ArgumentParser(description="Image-Classifier Project")
    parser.add_argument("input")
    parser.add_argument('checkpoint', action='store', default='checkpoint.pth')
    parser.add_argument('--top_k', dest='top_k', default='5')
    parser.add_argument('--filepath', dest='filepath', default=None)
    parser.add_argument('--category_names', dest='category_names', default='cat_to_name.json')
    parser.add_argument('--gpu', action='store_true', default=True)
    return parser.parse_args()



def main(): 
    args = todo_args()
    gpu = args.gpu
    model = load_checkpoint(args.checkpoint)
    cat_to_name = load_cat_to_name(args.category_names)
    
    if args.filepath == None:
        img_num = random.randint(1, 102)
        image = random.choice(os.listdir('./flowers/test/' + str(img_num) + '/'))
        img_path = './flowers/test/' + str(img_num) + '/' + image
        prob, classes = predict(img_path, model, int(args.top_k), gpu)
        print('Image:' + str(cat_to_name[str(img_num)]))
    else:
        img_path = args.filepath
        prob, classes = predict(img_path, model, int(args.top_k), gpu)
        print('File:'  + img_path)
    
    print(prob)
    print(classes)
    print([cat_to_name[x] for x in classes])

if __name__ == "__main__":
    main()
