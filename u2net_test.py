import os
from skimage import transform
import io 
import torch
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms#, utils
# import torch.optim as optim
import base64
import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB


def prepare_image(buffer: bytes) -> np.ndarray:
    
    
    image = Image.open(io.BytesIO(buffer)).convert("RGB")
    return np.array(image)


def apply_mask_and_save_with_output(np_image: np.ndarray, mask: np.ndarray, output_path: str) -> str:
    image_rgba = np.dstack((np_image, mask))

    result = Image.fromarray(image_rgba, mode="RGBA")

    result.save(output_path)

    buffer = io.BytesIO()
    result.save(buffer, format="PNG")
    # buffer.seek(0)
    
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def generateMask(original_image: np.ndarray, pred: torch.Tensor) -> np.ndarray:
       
    predict = pred.squeeze()
    predict_np = predict.cpu().data.numpy()
   
    im = Image.fromarray((predict_np * 255).astype(np.uint8)).convert('L')
   
    imo = im.resize((original_image.shape[1], original_image.shape[0]), resample=Image.BILINEAR)
  
    mask_array = np.array(imo)
   
    return mask_array

def main(image_buffer, save_dir):

    # --------- 1. get image path and name ---------
    model_name='u2net'#u2netp

    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name, model_name + '.pth')

    # --------- 2. dataloader ---------
    
    np_image = prepare_image(image_buffer)
 
    test_salobj_dataset = SalObjDataset(img_data_list = [np_image],
                                       
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
   
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)
    
    # --------- 3. model define ---------
    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)
    elif(model_name=='u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3,1)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()
    

    # --------- 4. inference for each image ---------
    for i_test, data_test in enumerate(test_salobj_dataloader):

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)
        
        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)
      
        
        mask = generateMask(np_image,pred)
        
        result = apply_mask_and_save_with_output(np_image, mask, './test_data/u2net_results/output.png')
        return result

        del d1,d2,d3,d4,d5,d6,d7

if __name__ == "__main__":
    main()
