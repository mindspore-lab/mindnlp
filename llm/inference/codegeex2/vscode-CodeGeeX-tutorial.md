# **Visual Studio code CodeGeeX插件操作指南**

## 1. **CodeGeeX安装**

1. 通过如下链接安装Visual Studio Code（https://code.visualstudio.com）

2. 打开Visual Studio Code，按照下图所示，点击左侧工具栏中的extensions图标，搜索CodeGeeX，最后点击install进行安装

![image](https://github.com/user-attachments/assets/41e436ed-3877-464b-9785-45f1b728b21d)


3. 安装成功后，左侧工具栏及右下角会显示CodeGeeX的图标，如下图所示

![image](https://github.com/user-attachments/assets/9f8706c2-bc6d-42df-9f0e-96cbd833e229)


## 2. **CodeGeeX功能介绍**

CodeGeeX具备如下功能：

1. 代码生成与补全：输入代码或描述函数的注释，CodeGeeX会自动生成后续代码
2. 代码翻译：将选定代码段翻译成其他编程语言
3. 注释生成：为选定代码段补充注释
4. 测试生成：根据代码生成unittest
5. 智能问答：由CodeGeeX针对用户输入的代码编程问题进行答复

如下我们将逐个演示CodeGeeX的功能。

### 2.1 **代码生成与补全**

1. 点击左侧的Explorer，后点击open folder，打开指定文件夹（本操作选择名为CodeGeeX_demo的文件夹进行演示）

![image](https://github.com/user-attachments/assets/b9433c9a-f5fc-4d4b-83b4-be731c5a7f07)


2. 点击图中所示图标，新建文件，并命名为code_generation.py

![image](https://github.com/user-attachments/assets/ac4dbb47-3abc-4293-80ad-1d8e53e14d36)


本次操作中展示如何让CodeGeeX生成冒泡排序函数

3. 在文件中输入注释：# write a bubble sort function，CodeGeeX生成代码，点击tab键接受生成内容

![image](https://github.com/user-attachments/assets/951bf370-f331-4806-b131-32aaba415fa8)


4. 如下图所示，CodeGeeX会逐步生成代码

![image](https://github.com/user-attachments/assets/8ee02163-da5b-426f-98ea-0c49e3ca18e5)

![image](https://github.com/user-attachments/assets/a09e36e8-85f8-484d-8687-d2a2b2f5e6f6)

![image](https://github.com/user-attachments/assets/d6f73055-626f-47b6-a7b7-0ee62e790b0a)

## 2.2 **代码翻译**

 

1. 选中上个操作生成的代码，点击Ctrl+Alt+T进入翻译模式，本次示例中我们尝试将python代码翻译为c++代码

![image](https://github.com/user-attachments/assets/6c352784-947e-44aa-a376-53d9ffb53cbb)

2. 在Translate into中，选择希望翻译成的语言，此处我们选择C++，然后点击Translate

![image](https://github.com/user-attachments/assets/d0cb09a0-2e3e-488e-be4f-c6d99a410d7e)

3. CodeGeeX在Output Code中生成C++代码

![image](https://github.com/user-attachments/assets/5c1cbe7a-9c31-419d-9200-e7afb62d6176)

## 2.3 测试生成

1. 选中生成的冒泡排序函数，点击右键，选择CodeGeeX-Generate Tests

![image](https://github.com/user-attachments/assets/b733819f-77c3-4cfa-b296-97139006e696)

2. CodeGeeX会在左侧生成对该函数的unittest

![image](https://github.com/user-attachments/assets/2db94c0f-f278-4030-a838-2c22158ca51c)

## 2.4 注释生成

1. 删除原代码中的注释，选中代码并单击右键，选择CodeGeeX-Generate Comment

![image](https://github.com/user-attachments/assets/41fd5f5c-87e4-4868-998c-03227c38dcca)
 

2. CodeGeeX可以生成中文或英文注释，本次演示中我们选择英文注释

![image](https://github.com/user-attachments/assets/eb25ead1-11f3-4565-8dd0-a0624c8164ae)

3. CodeGeeX生成注释

![image](https://github.com/user-attachments/assets/85885bf1-5b32-48f1-aad8-9c5b803d4938)

## 2.5 智能问答

1. 选中代码，点击Ctrl+Alt+I进入智能问答界面

![image](https://github.com/user-attachments/assets/edd86c4b-f7b2-4b06-aa4e-0ad48ab0d0a4)

2. 输入问题，如：帮我解释这段代码

![image](https://github.com/user-attachments/assets/914fa53a-bc42-4c18-9708-facf4c93de11)


3. CodeGeeX生成对代码的解析

![image](https://github.com/user-attachments/assets/c47524c0-3ec2-40c3-b1da-2b4ba9c4d8c4)
