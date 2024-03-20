import datetime
import math
import unittest
import mindspore
import random

from mindnlp.transformers import MSChatGLMForConditionalGeneration, AutoTokenizer
from mindnlp.utils.testing_utils import require_mindspore, slow

from mindspore._c_expression import _framework_profiler_step_start
from mindspore._c_expression import _framework_profiler_step_end
# mindspore.set_context(mode=mindspore.GRAPH_MODE, jit_syntax_level=mindspore.LAX, save_graphs=True, save_graphs_path="./graph")

def set_random_seed(seed):
    import random

    random.seed(seed)

    # pytorch RNGs
    import mindspore

    mindspore.set_seed(seed)

    # numpy RNG
    import numpy as np

    np.random.seed(seed)



def ids_tensor(shape, vocab_size):
    #  Creates a random int32 tensor of the shape within the vocab size
    total_dims = 1
    for dim in shape:
        total_dims *= dim

    values = []
    for _ in range(total_dims):
        values.append(random.randint(0, vocab_size - 1))

    return mindspore.tensor(data=values, dtype=mindspore.int64).view(shape)


def get_model_and_tokenizer():
    model = MSChatGLMForConditionalGeneration.from_pretrained("THUDM/chatglm-6b").half()
    model.set_train(False)
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b")
    return model, tokenizer

@require_mindspore
class ChatGLMGenerationTest(unittest.TestCase):
    def get_generation_kwargs(self):
        pass

    @slow
    def test_chat(self):
        model, tokenizer = get_model_and_tokenizer()
        prompts = ["你好", "介绍一下清华大学", "它创建于哪一年"]
        history = []
        set_random_seed(42)
        expected_responses = [
            '你好👋！我是人工智能助手 ChatGLM-6B，很高兴见到你，欢迎问我任何问题。',
            '清华大学是中国著名的综合性研究型大学，位于中国北京市海淀区，创建于 1911 年，前身是清华学堂。作为我国顶尖高等教育机构之一，清华大学在科学研究、工程技术、信息技术、经济管理等领域处于领先地位，也是世界上最著名的工程学府之一。\n\n清华大学拥有世界一流的教学设施和科学研究平台，设有多个学院和研究中心，包括工程学院、自然科学学院、社会科学学院、人文学院、法学院、经济管理学院等。学校拥有众多知名教授和研究团队，其中包括多位院士、国家杰出青年科学基金获得者、长江学者等。\n\n清华大学的本科生招生范围为全国中学毕业生，本科生入学要求严格，考试成绩优秀。同时，清华大学也提供研究生和博士生招生，包括硕士研究生和博士研究生。',
            '清华大学创建于 1911 年。'
        ]
        for (prompt, expected_response) in zip(prompts, expected_responses):
            _framework_profiler_step_start()
            response, history = model.chat(tokenizer, prompt, history=history)
            _framework_profiler_step_end()
            print(repr(response))
            break
            self.assertEquals(expected_response, response)

    @slow
    def test_stream_chat(self):
        model, tokenizer = get_model_and_tokenizer()
        prompts = ["你好", "介绍一下清华大学", "它创建于哪一年"]
        history = []
        expected_responses = [
            '你好👋！我是人工智能助手 ChatGLM-6B，很高兴见到你，欢迎问我任何问题。',
            '清华大学是中国著名的综合性研究型大学，位于中国北京市海淀区，创建于 1911 年，前身是清华学堂。作为我国顶尖高等教育机构之一，清华大学在科学研究、工程技术、信息技术、经济管理等领域处于领先地位，也是世界上最著名的工程学府之一。\n\n清华大学拥有世界一流的教学设施和科学研究平台，设有多个学院和研究中心，包括工程学院、自然科学学院、社会科学学院、人文学院、法学院、经济管理学院等。学校拥有众多知名教授和研究团队，其中包括多位院士、国家杰出青年科学基金获得者、长江学者等。\n\n清华大学的本科生招生范围为全国中学毕业生，本科生入学要求严格，考试成绩优秀。同时，清华大学也提供研究生和博士生招生，包括硕士研究生和博士研究生。',
            '清华大学创建于 1911 年。'
        ]
        set_random_seed(42)
        for prompt, expected_response in zip(prompts, expected_responses):
            response = ""
            for idx, (response, history) in enumerate(model.stream_chat(tokenizer, prompt, history=history)):
                pass
            print(repr(response))
            self.assertEquals(expected_response, response)

    @slow
    def test_generation(self):
        model, tokenizer = get_model_and_tokenizer()
        sentence = "晚上睡不着怎么办"
        parameters = [(False, 2048, 1),
                      (False, 64, 1),
                      (True, 2048, 1),
                      (True, 64, 1),
                      (True, 2048, 4)]
        expected_out_sentences = [
            '晚上睡不着怎么办 以下是一些可能有助于在晚上入睡的方法:\n\n1. 保持规律的睡眠时间表:尽量在同一时间上床,并尝试在早上醒来时自然起床。\n\n2. 创建舒适的睡眠环境:保持房间安静、凉爽、黑暗、舒适,并使用舒适的床垫和枕头。\n\n3. 避免刺激性物质:避免饮用含咖啡因的饮料,如咖啡、茶和可乐,并尽可能减少饮酒。\n\n4. 放松身心:尝试进行放松的活动,如冥想、深呼吸、瑜伽或听轻柔的音乐。\n\n5. 避免在床上做其他事情:例如看电视、使用电脑或智能手机等。\n\n6. 练习放松技巧:例如渐进性肌肉松弛法、冥想或深呼吸练习。\n\n7. 寻求帮助:如果长时间都无法正常入睡,可以考虑咨询医生或专业心理医生,寻求更进一步的帮助。\n\n希望这些方法能有助于入睡。',
            '晚上睡不着怎么办 以下是一些可能有助于在晚上入睡的方法:\n\n1. 保持规律的睡眠时间表:尽量在同一时间上床,并尝试在早上醒来时自然起床。\n\n2. 创建舒适的睡眠环境:保持房间安静、凉爽、黑暗、舒适,并使用舒适的床垫和枕头。',
            '晚上睡不着怎么办 以下是一些有助于在晚上更好地入睡的方法:\n\n1. 维持规律的睡眠时间:每晚尽可能在同一时间上床,保持规律的睡眠时间表,帮助身体调整并更容易入睡。\n\n2. 避免在床上使用电子设备:手机、平板电脑、电脑等电子设备会发出蓝光,这会干扰身体释放褪黑素,进而导致难以入睡。建议你在睡前一小时停止使用这些设备。\n\n3. 创建舒适的睡眠环境:确保卧室安静、黑暗、凉爽,舒适的床垫和枕头,保持卧室温度适宜,这有助于让你更容易入睡。\n\n4. 放松身心:尝试进行一些放松的活动,如冥想、深呼吸、瑜伽或轻松的散步,减轻压力和焦虑,让你更容易入睡。\n\n5. 避免咖啡因和酒精:咖啡因和酒精会让大脑更加兴奋,进而干扰身体入睡过程。建议在睡前几小时避免饮用这些物质。\n\n6. 做一些安静的活动:阅读一本书、听轻柔的音乐、绣或者绘画等安静的活动,有助于自己放松身心,进而更容易入睡。\n\n如果采取以上这些方法仍然无法入睡,建议咨询医生或专业的睡眠专家,获取更好的建议和帮助。',
            '晚上睡不着怎么办 以下是一些有助于在晚上更好地入睡的方法:\n\n1. 维持规律的睡眠时间:每晚尽可能在同一时间上床,保持规律的睡眠时间表,帮助身体调整并更容易入睡。\n\n2. 避免在床上使用电子设备:手机、平板电脑、电脑等电子设备会发出蓝光,这会干扰身体',
            '晚上睡不着怎么办 以下是一些可能有助于在晚上入睡的方法:\n\n1. 建立规律的睡眠时间表:尽量在同一时间入睡和起床,即使在周末和假期也要尽量保持一致。\n\n2. 创造舒适的睡眠环境:保持房间安静、凉爽、黑暗、舒适,使用舒适的床垫和枕头等。\n\n3. 放松身心:尝试进行一些放松的活动,如冥想、深呼吸、瑜伽、听轻柔的音乐等,缓解压力和紧张情绪。\n\n4. 避免刺激性物质:避免饮用咖啡、茶、可乐等含咖啡因的饮料,避免吸烟和饮酒等刺激性物质。\n\n5. 避免躺在床上翻来覆去:如果躺在床上超过20分钟还不能入睡,就不要躺在床上翻来覆去,而是起床去做一些放松的活动,直到感到困倦为止。\n\n6. 练习放松技巧:如果感到焦虑或紧张,可以尝试进行一些放松技巧,如渐进性肌肉松弛、冥想等。\n\n7. 改善睡眠障碍:如果已经尝试了上述方法仍然无法入睡,可以考虑咨询医生,了解是否存在其他睡眠障碍问题,并接受相应的治疗。']
        for (do_sample, max_length, num_beams), expected_output_sentence in zip(parameters, expected_out_sentences):
            set_random_seed(42)
            inputs = tokenizer(sentence, return_tensors="ms")

            outputs = model.generate(
                **inputs,
                do_sample=do_sample,
                max_length=max_length,
                num_beams=num_beams
            )

            outputs = outputs.tolist()[0]
            out_sentence = tokenizer.decode(outputs, skip_special_tokens=True)
            print(out_sentence)
            self.assertEquals(expected_output_sentence, out_sentence)

    @slow
    def test_batch_generation(self):
        model, tokenizer = get_model_and_tokenizer()
        sentences = [
            "你好",
            "介绍一下清华大学"
        ]
        parameters = [
                      (False, 2048, 1),
                      (False, 64, 1),
                      (True, 2048, 1),
                      (True, 64, 1),
                      (True, 2048, 4)]
        expected_out_sentences = [
            ['你好 你好👋!我是人工智能助手 ChatGLM-6B,很高兴见到你,欢迎问我任何问题。',
             '介绍一下清华大学 清华大学是中国著名的综合性大学,位于北京市海淀区双清路30号,其历史可以追溯到1911年创建的清华学堂,1925年更名为清华学校,1937年抗日战争全面爆发后南迁长沙,1946年迁回清华园。新中国成立后,清华学校更名为清华大学。\n\n清华大学是中国最顶尖的大学之一,在工程、科学、技术、经济、管理等领域都有很高的学术声誉和影响力。学校拥有世界一流的教学设施和科学研究平台,有多个学院和研究中心,包括工程学院、自然科学学院、人文学院、社会科学学院、经济管理学院、法学院、美术学院、医学院、器学院等。\n\n清华大学的本科生招生始于2000年,实行全面二孩政策后,本科生招生规模不断扩大。截至2022年,清华大学共有本科生近3万人,研究生近2万人,其中国际学生占比约为10%。清华大学的本科生教育注重通识教育和个性化培养,强调实践、创新、国际化和综合素质。'],
            [
                '你好 你好👋!我是人工智能助手 ChatGLM-6B,很高兴见到你,欢迎问我任何问题。',
                '介绍一下清华大学 清华大学是中国著名的综合性大学,位于北京市海淀区双清路30号,其历史可以追溯到1911年创建的清华学堂,1925年更名为清华学校,1937年抗日战争全面爆发后南迁长沙,1946年迁回'
            ],
            [
                '你好 你好👋!我是人工智能助手 ChatGLM-6B,很高兴见到你,欢迎问我任何问题。',
                '介绍一下清华大学 清华大学是中国著名的综合性研究型大学,位于北京市海淀区双清路 30 号,其溯源于 1911 年创建的清华学堂, 1925 年更名为清华学校, 1937 年秋抗日战争全面爆发后闭校。1949 年 10 月开学复校,成为我国第一个社会主义大学生活了的高校。截至 2023 年,清华学校共管辖 2 个学院、13 个系,有本科专业 60 个,研究生专业 190 个。'
            ],
            [
                '你好 你好👋!我是人工智能助手 ChatGLM-6B,很高兴见到你,欢迎问我任何问题。',
                '介绍一下清华大学 清华大学是中国著名的综合性研究型大学,位于北京市海淀区双清路 30 号,其溯源于 1911 年创建的清华学堂, 1925 年更名为清华学校, 1937 年秋抗日战争全面爆发后'
            ],
            [
                '你好 你好👋!我是人工智能助手 ChatGLM-6B,很高兴见到你,欢迎问我任何问题。',
                '介绍一下清华大学 清华大学是中国著名的综合性研究型大学,位于北京市海淀区双清路30号,其历史可以追溯到1911年创建的清华学堂,1925年更名为清华学校,1937年抗日战争全面爆发后南迁长沙,与北京大学、南开大学组建国立长沙临时大学,1938年迁至 昆明改名为国立西南联合大学,1946年迁回北京。新中国成立后,清华学校更名为清华大学。'
            ]
        ]
        for (do_sample, max_length, num_beams), expected_output_sentence in zip(parameters, expected_out_sentences):
            set_random_seed(42)
            inputs = tokenizer(sentences, return_tensors="ms", padding=True)

            outputs = model.generate(
                **inputs,
                do_sample=do_sample,
                max_length=max_length,
                num_beams=num_beams
            )

            batch_out_sentence = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            print(batch_out_sentence)
            self.assertListEqual(expected_output_sentence, batch_out_sentence)
