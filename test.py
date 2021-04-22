import os
from PIL import Image
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from model import resnet_cbam

DATA_ROOT = './testimg'
RESULT_FILE = 'result.csv'

def test_and_generate_result_round2(epoch_num='1', model_name='resnet101_cbam', img_size=320):
    data_transform = transforms.Compose([
        transforms.Resize(img_size, Image.ANTIALIAS),
        transforms.ToTensor(),
        transforms.Normalize([0.53744068, 0.51462684, 0.52646497], [0.06178288, 0.05989952, 0.0618901])
    ])

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    is_use_cuda = torch.cuda.is_available()

    my_model = torch.load('./savemodel/' + model_name + '/Models_epoch_' + epoch_num + '.pkl')


    if is_use_cuda:
        my_model = my_model.cuda()
    my_model.eval()


    with open(os.path.join('savemodel', model_name, model_name+'_'+str(img_size)+'_'+RESULT_FILE), 'w', encoding='utf-8') as fd:
        fd.write('filename|defect,probability\n')
        test_files_list = os.listdir(DATA_ROOT)
        for _file in test_files_list:
            file_name = _file
            if '.jpg' not in file_name:
                continue
            file_path = os.path.join(DATA_ROOT, file_name)
            img_tensor = data_transform(Image.open(file_path).convert('RGB')).unsqueeze(0)

            if is_use_cuda:
                    img_tensor = Variable(img_tensor.cuda(), volatile=True)

            output = my_model(img_tensor)
            output = F.softmax(output, dim=1)
            print(output)


if __name__ == '__main__':

    test_and_generate_result_round2()
