# **Visual Studio code CodeGeeX插件操作指南**

## 1. **CodeGeeX安装**

1. 通过如下链接安装Visual Studio Code（https://code.visualstudio.com）

2. 打开Visual Studio Code，按照下图所示，点击左侧工具栏中的extensions图标，搜索CodeGeeX，最后点击install进行安装

![image-20240827164543278](C:\Users\10205\AppData\Roaming\Typora\typora-user-images\image-20240827164543278.png) 

3. 安装成功后，左侧工具栏及右下角会显示CodeGeeX的图标，如下图所示

![image-20240827164555271](C:\Users\10205\AppData\Roaming\Typora\typora-user-images\image-20240827164555271.png) 

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

 ![image-20240827164612752](C:\Users\10205\AppData\Roaming\Typora\typora-user-images\image-20240827164612752.png) 

2. 点击图中所示图标，新建文件，并命名为code_generation.py

 ![image-20240827164629984](C:\Users\10205\AppData\Roaming\Typora\typora-user-images\image-20240827164629984.png) 

本次操作中展示如何让CodeGeeX生成冒泡排序函数

3. 在文件中输入注释：# write a bubble sort function，CodeGeeX生成代码，点击tab键接受生成内容

![img](file:///C:\Users\10205\AppData\Local\Temp\ksohtml6752\wps5.jpg) 

4. 如下图所示，CodeGeeX会逐步生成代码

![img](file:///C:\Users\10205\AppData\Local\Temp\ksohtml6752\wps6.jpg) 

![img](file:///C:\Users\10205\AppData\Local\Temp\ksohtml6752\wps7.jpg) 

![img](file:///C:\Users\10205\AppData\Local\Temp\ksohtml6752\wps8.jpg) 

## 2.2 **代码翻译**

 

1. 选中上个操作生成的代码，点击Ctrl+Alt+T进入翻译模式，本次示例中我们尝试将python代码翻译为c++代码

 ![image-20240827164649745](C:\Users\10205\AppData\Roaming\Typora\typora-user-images\image-20240827164649745.png) 

2. 在Translate into中，选择希望翻译成的语言，此处我们选择C++，然后点击Translate

![image-20240827164700680](C:\Users\10205\AppData\Roaming\Typora\typora-user-images\image-20240827164700680.png) 

3. CodeGeeX在Output Code中生成C++代码

![image-20240827164715664](C:\Users\10205\AppData\Roaming\Typora\typora-user-images\image-20240827164715664.png) 

## 2.3 测试生成

1. 选中生成的冒泡排序函数，点击右键，选择CodeGeeX-Generate Tests

 ![image-20240827164809102](C:\Users\10205\AppData\Roaming\Typora\typora-user-images\image-20240827164809102.png) 

 

2. CodeGeeX会在左侧生成对该函数的unittest

![image-20240827164828538](C:\Users\10205\AppData\Roaming\Typora\typora-user-images\image-20240827164828538.png) 

## 2.4 注释生成

1. 删除原代码中的注释，选中代码并单击右键，选择CodeGeeX-Generate Comment

![img](file:///C:\Users\10205\AppData\Local\Temp\ksohtml6752\wps14.jpg) 

 

2. CodeGeeX可以生成中文或英文注释，本次演示中我们选择英文注释

![image-20240827164848672](C:\Users\10205\AppData\Roaming\Typora\typora-user-images\image-20240827164848672.png) 

3. CodeGeeX生成注释

 ![image-20240827164900388](C:\Users\10205\AppData\Roaming\Typora\typora-user-images\image-20240827164900388.png) 

## 2.5 智能问答

1. 选中代码，点击Ctrl+Alt+I进入智能问答界面

![image-20240827164915094](C:\Users\10205\AppData\Roaming\Typora\typora-user-images\image-20240827164915094.png) 

2. 输入问题，如：帮我解释这段代码

![image-20240827164928198](C:\Users\10205\AppData\Roaming\Typora\typora-user-images\image-20240827164928198.png) 

3. CodeGeeX生成对代码的解析

![image-20240827164944202](C:\Users\10205\AppData\Roaming\Typora\typora-user-images\image-20240827164944202.png) 