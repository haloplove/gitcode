import tkinter as tk
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import os

# 确定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#定义超参数
BATCH_SIZE=128
DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#cpu训练或者gpu训练
EPOCHS=10#训练次数

#构建网络模型
class Digit(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(1,10,5)#灰度图片通道， 输出通道，卷积核大小
        self.conv2=nn.Conv2d(10,20,3)
        self.fcl=nn.Linear(20*10*10,500)
        self.fc2=nn.Linear(500,10)#输入通道，输出通道

    def forward(self,x):
        input_size=x.size(0)
        x=self.conv1(x)
        x=F.relu(x)
        x=F.max_pool2d(x,2,2)

        x=self.conv2(x)
        x=F.relu(x)

        x=x.view(input_size,-1)#数据降维

        x=self.fcl(x)
        x=F.relu(x)

        x=self.fc2(x)

        output=F.log_softmax(x,dim=1)

        return output




# 创建一个 280x280 的画板，然后我们可以在保存图像时将其缩放到 28x28
width, height = 280, 280
center = height//2
white = (255, 255, 255)
green = (0,128,0)
black = (0, 0, 0)  # 修改为黑色

def save():
    # 将画板内容保存为 PIL 图像
    image = Image.new("RGB", (width, height), white)
    draw = ImageDraw.Draw(image)
    for line in lines:
        draw.line(line, fill=black, width=10)
    # 缩放图像到 28x28
    image = image.resize((28, 28))
    # 保存图像
    image.save('output.png')

def paint(event):
    x1, y1 = (event.x - 1), (event.y - 1)
    x2, y2 = (event.x + 1), (event.y + 1)
    cv.create_oval(x1, y1, x2, y2, fill="black",width=5)
    lines.append([x1, y1, x2, y2])

root = tk.Tk()

# 创建一个画板
cv = tk.Canvas(root, width=width, height=height, bg='white')
cv.pack()

# 存储所有的线条
lines = []

cv.pack(expand=1, fill="both")
cv.bind("<B1-Motion>", paint)

# 添加一个保存按钮
button = tk.Button(text="save", command=save)
button.pack()

root.mainloop()



model=Digit().to(DEVICE)

# 加载模型参数
model.load_state_dict(torch.load('model_parameters.pth'))



# 定义图像预处理
preprocess = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 指定要识别的图像文件
image_file = "output.png"

if os.path.exists(image_file):
    image = Image.open(image_file).convert('L')


    # 对图像进行预处理
    image = preprocess(image)
     # 添加批次维度并将其移动到正确的设备上
    image = image.unsqueeze(0).to(device)
    # 我们将图像传递给模型进行预测
    output = model(image)

# 模型的输出是一个包含 10 个元素的向量，每个元素表示图像是对应数字的概率
# 我们可以使用 argmax 来找到最大概率的索引，这就是模型的预测
    prediction = output.argmax(dim=1).item()
    print('Predicted number:', prediction)


