import scipy
import mindspore
from IPython.display import Audio
from mindnlp.transformers.models.bark import BarkModel, BarkProcessor


voice_preset = None
def main():
    print("欢迎使用 Bark模型,输入下列任一数字选择你所需要的模型规模,或者输入stop提前终止程序")
    print("------------------------------------------------------------")
    print("|                 1. bark-small                            |")
    print("|                 2. bark-large                            |") 
    print("------------------------------------------------------------")   
    print("注意: 如果你希望使用不同的说话人模式, 由于相应的模型无法直接下载")
    print("请自行下载后,并修改voice_preset使其指向文件对应的位置")
    choose = input("请输入你的选择:")
    if (choose == "stop"):
        return 0
    else:
        if(choose != "1" and choose != "2"):
            print("选择无效,即将退出")
            return 0
        Processor = BarkProcessor.from_pretrained("suno/bark-small") if choose=="1" else BarkProcessor.from_pretrained("suno/bark")
        Model = BarkModel.from_pretrained("suno/bark-small") if choose=="1" else BarkModel.from_pretrained("suno/bark")
        Model.set_train(False)
    while True:
        inputs = input("请输入你想要让我说的话(可以带上大笑[laugh]等语气词):")
        if inputs == "stop":
            return 0
        inputs = Processor(inputs, voice_preset = voice_preset)
        audio_array = Model.generate(**inputs,pad_token_id=10)
        audio_array = audio_array.numpy().squeeze()
        sample_rate = Model.generation_config.sample_rate
        Audio(audio_array, rate=sample_rate, autoplay=True)
        scipy.io.wavfile.write("bark_out_ms.wav", rate=sample_rate, data=audio_array)

if __name__ == "__main__":
    main() 