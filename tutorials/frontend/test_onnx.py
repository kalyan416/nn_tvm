import onnx
import numpy as np
import tvm
from tvm import te
import tvm.relay as relay
from tvm.contrib.download import download_testdata


onnx_model = onnx.load('/home/kalyan/models/onnx/resnet18/resnet18.onnx')

from PIL import Image
img_url = 'https://github.com/dmlc/mxnet.js/blob/master/data/cat.png?raw=true'
img_path = download_testdata(img_url, 'cat.png', module='data')
img = Image.open(img_path).resize((224, 224))

# Preprocess the image and convert to tensor
from torchvision import transforms
my_preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
img = my_preprocess(img)
img = np.expand_dims(img, 0)
x = img

target = 'llvm'

input_name = 'input.1'
shape_dict = {input_name: x.shape}
mod, params = relay.frontend.from_onnx(onnx_model, shape_dict)

with relay.build_config(opt_level=1):
    intrp = relay.build_module.create_executor('graph', mod, tvm.cpu(0), target)

######################################################################
# Execute on TVM
# ---------------------------------------------
dtype = 'float32'
tvm_output = intrp.evaluate()(tvm.nd.array(x.astype(dtype)), **params).asnumpy()

synset_url = ''.join(['https://raw.githubusercontent.com/Cadene/',
                      'pretrained-models.pytorch/master/data/',
                      'imagenet_synsets.txt'])
synset_name = 'imagenet_synsets.txt'
synset_path = download_testdata(synset_url, synset_name, module='data')
with open(synset_path) as f:
    synsets = f.readlines()

synsets = [x.strip() for x in synsets]
splits = [line.split(' ') for line in synsets]
key_to_classname = {spl[0]:' '.join(spl[1:]) for spl in splits}

class_url = ''.join(['https://raw.githubusercontent.com/Cadene/',
                      'pretrained-models.pytorch/master/data/',
                      'imagenet_classes.txt'])
class_name = 'imagenet_classes.txt'
class_path = download_testdata(class_url, class_name, module='data')
with open(class_path) as f:
    class_id_to_key = f.readlines()

class_id_to_key = [x.strip() for x in class_id_to_key]

# Get top-1 result for TVM
top1_tvm = np.argmax(tvm_output[0])
tvm_class_key = class_id_to_key[top1_tvm]
print('Relay top-1 id: {}, class name: {}'.format(top1_tvm, key_to_classname[tvm_class_key]))
