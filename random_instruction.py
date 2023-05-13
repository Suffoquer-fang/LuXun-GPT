import argparse
import json
from tqdm import tqdm
import random

def format_example(example: dict) -> dict:
    instructions = ["将这句话改写成鲁迅风格的语言", "用鲁迅的风格改写", "你是一个非常熟悉鲁迅风格的作家，请用鲁迅的风格改写这句话", "用鲁迅的风格改写这句话"]
    instruction = random.choice(instructions)
    context = f"Instruction: {instruction}\n"
    if example.get("Input"):
        context += f"Input: {example['Input']}\n"
    context += "Answer: "
    target = example["Output"]
    return {"context": context, "target": target}

        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="example_data/selected_aug.jsonl")
    parser.add_argument("--save_path", type=str, default="example_data/luxun_data.jsonl")

    args = parser.parse_args()

    print("#> Formatting dataset...")
    print("#> Input path: {}".format(args.data_path))
    print("#> Output path: {}".format(args.save_path))

    random.seed(42)

    if args.data_path.endswith(".json"):
        with open(args.data_path) as f:
            examples = json.load(f)
    elif args.data_path.endswith(".jsonl"):
        examples = []
        with open(args.data_path) as f:
            for line in f:
                examples.append(json.loads(line))

    with open(args.save_path, 'a+') as f:
        for example in tqdm(examples, desc="formatting.."):
            datapoint = format_example(example)
            f.write(json.dumps(datapoint, ensure_ascii=False) + '\n')

    print("#> Formatting finished!", "Total examples:", len(examples))

if __name__ == "__main__":
    main()
