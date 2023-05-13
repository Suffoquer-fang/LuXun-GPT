# LuXun-GPT: ChatGLM+LoRA微调的鲁迅风格改写模型

本项目开源了经过LoRA指令微调(Instruct-tuning)的ChatGLM-6B模型，可以用鲁迅的语言风格改写给定的输入。

针对给定的任务（鲁迅风格的文本改写/翻译），通过鲁迅的文章和GPT3.5 API构建了对应的指令数据集，并在此基础上对ChatGLM-6B进行了指令微调，提高了具体任务的效果。

## Quick Start

要使用该模型，您可以按照以下步骤进行：

1. 克隆仓库：

```
git clone https://github.com/Suffoquer-fang/LuXun-GPT.git
cd LuXun-GPT
```

2. 安装所需的依赖：

```
pip install -r requirements.txt
```

3. 下载训练好的LoRA参数：[Huggingface](https://huggingface.co/Suffoquer/LuXun-lora/tree/main)
```python
model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True, load_in_8bit=True, device_map='auto')
peft_model = PeftModel.from_pretrained(model, "Suffoquer/LuXun-lora")
```

4. 交互体验
```shell
CUDA_VISIBLE_DEVICES=0 \
python inference.py \
    --lora Suffoquer/LuXun-lora \
    --instruction 用鲁迅风格的语言改写，保持原来的意思： \
    --interactive
```

## 数据集构建

通过将鲁迅的文章反向改写为原始文本，可以构建pair-wise的训练数据。
```
{
    "Input": "在我的后园，可以看见墙外有两株树，一株是枣树，还有一株也是枣树。"
    "Output": "在我的院子里有两棵树，都是枣树。"
}
```

利用ChatGPT 3.5的接口，可以进行不同领域的数据扩增，使得最终的训练样本更加丰富。
```
{
    "Input": "在我的后园，可以看见墙外有两株树，一株是枣树，还有一株也是枣树。", 
    "Output": [
        "在博物馆里，可以看见有两件文物，一件是陶器，还有一件也是陶器。",
        "在公园里，可以看见有两只鸟，一只是麻雀，还有一只也是麻雀。",
        "在我的画室里，可以看见画架上有两幅画，一幅是油画，还有一幅也是油画。",
        "在我的邮箱里，可以看见两封信，一封是邀请函，还有一封也是邀请函。",
        "在电影院，可以看见有两部电影，一部是喜剧片，还有一部也是喜剧片。"
    ]
}
```

具体可见`crawl_prompt.py`

## 训练

数据预处理
```shell
cd scirpts
bash run_preprocess.sh
```

LoRA微调
```shell
CUDA_VISIBLE_DEVICES=0,1 \
python lora_finetune.py \
    --dataset_path example_data/luxun_dataset \
    --lora_rank 8 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --max_steps 5000 \
    --save_steps 1000 \
    --learning_rate 1e-4 \
    --fp16 \
    --remove_unused_columns false \
    --logging_steps 50 \
    --output_dir saved_models
```



## 推理

```shell
CUDA_VISIBLE_DEVICES=0 \
python inference.py \
    --lora Suffoquer/LuXun-lora \
    --instruction 用鲁迅风格的语言改写，保持原来的意思： \
    --input_path test_data/test.txt \
    --output_path test_data/output.txt 
```

你可以得到`output.txt`
```
Instruction: 用鲁迅风格的语言改写，保持原来的意思：

Input: 学习计算机技术，是没有办法救中国的。
Output: 学习计算机科学,根本没有办法救中国。

Input: 我怎么样都起不了床，我觉得我可能是得了抑郁症吧。
Output: 我至今仍然不起床,大约可以斗胆说,是得了抑郁症的缘故罢。

Input: 书桌上有两支笔，都是钢笔。
Output: 在书桌上,可以看见有两支笔,一支是钢笔,还有一支也是钢笔。
```

## 模型效果对比

| 测试输入 | ChatGLM输出 | ChatGPT-3.5输出 | LuXun-LoRA输出 |
|---|---|---|---|
| 学习计算机技术，是没有办法救中国的。 | 学习计算机技术,或许无法拯救中国,但正如鲁迅所说,我们不能放弃对知识的渴望,因为知识可以打开我们的视野,让我们看到更广阔的世界,并找到解决问题的方法。即使我们不能改变整个国家的命运,我们也可以通过不断学习和进步,为自己和后代创造更好的未来。 | 学习计算机技术，不能救国于水深火热之中。 | 学习计算机科学,根本没有办法救中国。 |
| 我怎么样都起不了床，我觉得我可能是得了抑郁症吧。 | 我欲起身,却床难动,自觉或许患有抑郁症。 | 哀哉斯人！訾于巳身，彼何自其？余盈觉昏沉，吾之思，此遽变矣，莫非是抑郁之兆？ | 我至今仍然不起床,大约可以斗胆说,是得了抑郁症的缘故罢。 |
| 书桌上有两支笔，都是钢笔。 | 桌子上放着两支笔,都是钢笔。一支是黑色的,一支是红色的。黑色的笔用来写日记,红色的笔用来签字。我觉得黑色的笔更加正式,而红色的笔更加鲜艳。所以,我会用黑色的笔写日记,而用红色的笔签字。 | 书斗中旁边低垂的挂笔架，望之，见有两枝钢笔。 | 在书桌上,可以看见有两支笔,一支是钢笔,还有一支也是钢笔。 |
| 我去买几个橘子，你就站在这里，不要走动。 | 尔等且慢走,吾在此等之。| 今之人，举步维艰，思虑犹豫，不自在于市井之中。然吾素好进田园，今午气正佳，遂欲觅数枚橘实，以慰闲情。尔当空守此地，勿行一步，犹如穿行于余心之中，无所不在也。 | 我倒不如去买几个橘子,让我站稳脚跟,你可以一直站在这里,不必走动。 |
| 落霞与孤鹜齐飞，秋水共长天一色。 | 落霞与孤鹜齐飞,秋水共长天一色,斯人独憔悴。 | 夕阳赤红，孤鹜独自飞翔，秋水波光粼粼，与长空交相辉映。 | 夕阳西下,孤独的鸟儿和霞光同飞,秋水天长,孤独的鸟儿和天空同色。 |
| 中国的首都是北京。| 中国之首都,当推北京。此城雄踞华夏,巨龙腾飞之地,其历史文化底蕴深厚,风景名胜美不胜收。城市居民朴实善良,交通便捷,美食佳肴更是享誉全球。然而,如今北京已非昔日之北京,现代化进程日新月异,吾辈当保持警觉,守护祖国之根本,让北京依旧保持其作为中国之首都的地位。| 北京，全国之首府也。 | 中国的政治的中心是在北京。 |

ChatGLM和ChatGPT很容易将鲁迅风格的语言理解为文言文，经过训练的模型表现会好一些，但是也容易出现一些曲解原意的情况。更多样例可以查看`test_data/output.txt`。

## 致谢
本项目参考了以下开源项目，在此对相关项目和研究开发人员表示感谢。

* ChatGLM: https://github.com/THUDM/ChatGLM-6B
* ChatGLM-Tuning: https://github.com/mymusise/ChatGLM-Tuning
* 数据集: https://github.com/Ac-heron/luxun

## 许可证

该项目基于 MIT 许可证发布 - 有关详情，请参见 [LICENSE](LICENSE) 文件。