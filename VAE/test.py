import torch
import torchvision
from network import Model
from PIL import Image


PATH='./state_dict.pth'
state_dict = torch.load(PATH)

model = Model()
model = torch.nn.DataParallel(model)
model.load_state_dict(state_dict)

sample = torch.randn(64, 10)
data = model.module.fc4(model.module.fc3(sample)).view(64, 10, 7, 7)
data = model.module.decoder(data)
torchvision.utils.save_image(data.view(64, 1, 28, 28), 'result2.png')


# sample = model.module.decoder(model.module.fc2(sample).view(64, 128, 7, 7)).cpu()
# torchvision.utils.save_image(sample.data.view(64, 1, 28, 28), 'result/sample_' + str(1) + '.png')
