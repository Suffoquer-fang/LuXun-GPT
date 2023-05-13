
import openai
openai.api_key = 'YOUR API KEY HERE'
import re, json
from tqdm import tqdm
import random

def return_random_prompt(text):
    system_prompt = "你需要仿照给定的句式，尽可能给出多样化的输出。我们将用于人工评估ChatGPT模型对指令的完成情况。要求:\n"

    # generate random topics
    topic_list = ["科技", "娱乐", "体育", "金融", "时政", "教育", "医疗", "旅游", "美食", "汽车", "房产", "文化", "历史", "地理", "自然", "人文", "社会", "法律", "军事", "政治", "经济", "文学", "艺术", "宗教", "哲学", "语言", "数学", "物理", "化学", "生物", "地球科学", "天文学", "计算机科学", "工程", "建筑", "设计", "音乐", "舞蹈", "电影", "电视", "动漫", "游戏", "健康", "美容", "时尚", "家居", "家电", "家具", "家装", "母婴", "育儿", "职场", "工作", "生活", "养生", "心理", "情感", "人际", "社交", "交友", "恋爱", "婚姻", "家庭", "亲子", "宠物", "动物", "植物", "食品", "饮料", "餐饮", "酒店", "购物", "消费", "理财", "税务", "法规", "法院", "司法", "刑事", "民事", "行政", "战争"]
    system_prompt += "1. 主题多样化，涵盖各个领域，例如：" + "、".join(random.sample(topic_list, 10)) + "等。\n"

    system_prompt += "2. 尽量和输入保持同样的语言风格。\n"
    system_prompt += "3. 使用“空气”替代“气氛”。\n"
    system_prompt += "4. 使用“罢”替代“吧”。\n"
    
    system_prompt += "Example 1:\n"
    system_prompt += "Input: 创作之可尊，想来翻译家该是知道的，然而他竟止于翻译者，一定因为他只能翻译，或者偏爱翻译的缘故。\n"
    system_prompt += "Output 1: "
    system_prompt += "绘画之可尊，想来艺术家该是知道的，然而他竟止于临摹者，一定因为他只能临摹，或者偏爱临摹的缘故。\n"
    system_prompt += "Output 2: "
    system_prompt += "科研之可尊，想来研究者该是知道的，然而他竟止于灌水者，一定因为他只能灌水，或者偏爱灌水的缘故。\n"
    
    system_prompt += "Example 2:\nInput: "
    system_prompt += "我横竖睡不着，仔细看了半夜，才从字缝里看出字来，满本都写着两个字是“吃人”\n"
    system_prompt += "Output 1: "
    system_prompt += "我横竖听不懂授课，对着课件仔细研究了半夜，才从字缝里看出字来，满本都写着两个字是“退学”\n"
    system_prompt += "Output 2: "
    system_prompt += "我横竖通不过科目一，仔细背了半夜，才从字缝里看出字来，满本都写着三个字是“不通过”\n"

    system_prompt += "Example 3:\nInput: "
    system_prompt += "在我的后园，可以看见墙外有两株树，一株是枣树，还有一株也是枣树。\n"
    system_prompt += "Output 1: "
    system_prompt += "在我的桌上，可以看见有两支笔，一支是钢笔，还有一支也是钢笔。\n"
    system_prompt += "Output 2: "
    system_prompt += "在楼下，站着两个外卖小哥，一个是美团外卖的，还有一个也是美团外卖的。\n"


    system_prompt += f"Input: {text}\n"
    system_prompt += "请给出满足条件的10条数据:\n"
    
    return system_prompt



def return_translate_prompt(text):
    system_prompt = "你需要用不同风格的语言改写输入，尽可能给出多样化的输出。要求:\n"

    system_prompt += "1. 与输入保持相同的含义，不要过多扩展\n"
    system_prompt += "2. 尽量用不同的语言风格，例如：正式文本风格，日常聊天，散文等。\n"

    system_prompt += "Example 1:\n"
    system_prompt += "Input：又如看见兵士打车夫，在先也要愤愤的，但现在也就转念道，倘使这车夫当了兵，这兵拉了车，大抵也就这么打，便再也不放在心上了。\n"
    system_prompt += "正式文本：再比如，假如我们看到兵士打车夫，本能地会感到愤怒，但是如果思考一下，假如这个车夫成了兵，那么车夫还是会被兵打，这样的情况就再也不会让我们感到愤怒了。\n"
    system_prompt += "日常对话：嗨，听说你见过战士打马车夫？我以前也很生气，但现在我想，如果他们身份交换一下，他还会这样打的吗？也许我们就不会那么在意了。\n"

    system_prompt += f"Input：{text}\n"
    system_prompt += "请给出满足条件的5条数据:\n"
    
    return system_prompt

def return_simple_prompt(text):
    system_prompt = "你需要用通俗易懂的语言改写输入。要求与输入保持相同的含义，不要过多扩展\n"

    system_prompt += f"Input：{text}\n"
    system_prompt += "请给出满足条件的5个不同的改写:\n"
    
    return system_prompt


def handle_data_augmentation(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",    # here we use `gpt-3.5-turbo` model, while Stanford-Alpaca uses `text-davinci-003`
        messages=[
            {"role": "user", "content": return_random_prompt(text)},
        ]
    )
    if "content" not in response["choices"][0]["message"]:
        return []
    msg = response["choices"][0]["message"]["content"]
    msg_list = msg.split('\n')
    msg_list = [msg for msg in msg_list if msg != '']
    msg_list = [msg[11:] if msg.startswith("Output 10:") else msg[10:] for msg in msg_list]
    return msg_list

def handle_simple(text):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",    # here we use `gpt-3.5-turbo` model, while Stanford-Alpaca uses `text-davinci-003`
        messages=[
            {"role": "user", "content": return_simple_prompt(text)},
        ]
    )
    if "content" not in response["choices"][0]["message"]:
        return []
    msg = response["choices"][0]["message"]["content"]
    msg_list = msg.split('\n')
    msg_list = [msg for msg in msg_list if msg != '']
    msg_list = [msg[3:] for msg in msg_list]
    return msg_list


if __name__ == "__main__":
    text = "我横竖睡不着，仔细看了半夜，才从字缝里看出字来，满本都写着两个字是“吃人”"
    print(handle_data_augmentation(text))
    

