import torch
import torchvision
from PIL import Image
from torch import nn

img_path = "../img/airplane.png"
img = Image.open(img_path)
img = img.convert('RGB')

transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((32, 32)),
    torchvision.transforms.ToTensor()
])
img = transform(img)
print(img.shape)

# 加载网络模型
test_model = torch.load("myModel_20.pth")
# print(test_model)


# 要求input张量为4阶(batch_size,Channel,Hight,Weight),而一般图片都是3阶的,需要reshape()
img = torch.reshape(img, (1, 3, 32, 32))
img = img.cuda()
test_model.eval()
with torch.no_grad():
    output = test_model(img)
print(output)
print(output.argmax(1))





