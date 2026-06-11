import torch
from transformers import Phi3ForCausalLM
import os
from matplotlib import pyplot as plt

from datasets import Dataset
from transformers import AutoTokenizer, set_seed
from trl import SFTTrainer
import json
import numpy as np

from peft import AutoPeftModelForCausalLM
from tqdm import tqdm


from transformers import GenerationMixin, DataCollatorWithPadding
import random

set_seed(42)


#Custom class for the adapted implementation of the xVal approach
class CustomXValPhi3forCausalLM(Phi3ForCausalLM, GenerationMixin):
    def __init__(self, config):
        super().__init__(config)
        self.num_token_id = 32002   #id of the numbers token in the tokenizers vocabulary
        self.num_token = "[NUM]"    #string representation of the numbers token

    #custom forward function to include the multiplication of the numbers array
    #including padding to match the size of the numbers to the input ids if necessary
    def forward(self, input_ids = None, numbers = None, **kwargs):
        inputs_embeds = self.get_input_embeddings()(input_ids)
        # print(inputs_embeds.dtype)
        if numbers is not None:
            # print(f"input_embeds.shape: {inputs_embeds.shape}")
            # print(f"numbers.shape: {numbers.shape}")

            if numbers.size(1) < inputs_embeds.size(1):
                pad_length =  inputs_embeds.size(1) - numbers.size(1)
                numbers = torch.cat([numbers, numbers.new_ones(numbers.size(0), pad_length)], dim=1)
            # print(f"numbers.shape: {numbers.shape}")
            numbers = numbers.unsqueeze(-1).to(dtype=inputs_embeds.dtype)
            # print(f"numbers.shape: {numbers.shape}")
            # print(numbers.dtype)
            inputs_embeds = numbers * inputs_embeds
        # print(inputs_embeds.dtype)
        # print(f"input_embeds.shape: {inputs_embeds.shape}")
        # print("")
        if "inputs_embeds" in kwargs: #removing input embeds and ids to ensure the model only uses the new input embeds
            kwargs.pop("inputs_embeds")
        if "input_ids" in kwargs:
            kwargs.pop("input_ids")

        outputs = super().forward(inputs_embeds = inputs_embeds, **kwargs)

        return outputs

    def prepare_inputs_for_generation(self, input_ids, numbers = None, **kwargs):
        model_input = {"input_ids": input_ids}
        if numbers is not None:
            model_input["numbers"] = numbers

        return model_input


# Custom data collator to include the new parameters (numbers) in the batching process for the training
class CustomDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.default_collator = DataCollatorWithPadding(self.tokenizer, padding='max_length') #DataCollatorWithPadding

    def __call__(self, features):
        input_ids = [feature["input_ids"] for feature in features]
        numbers = [feature["numbers"] for feature in features]
        labels = [feature["labels"] for feature in features]
        # print(input_ids)
        # print(numbers)
        # print(labels)
        # print(f"labels max: {max(labels, key=len)}")
        max_length_input = len(max(input_ids, key=len))
        max_length_number = len(max(numbers, key=len))
        max_length_label = len(max(labels, key=len))
        # print(max_length_input, max_length_number, max_length_label)
        max_length = max([max_length_label, max_length_input, max_length_number])

        self.default_collator.max_length = max_length
        batch = self.default_collator({"input_ids": input_ids}) #, "numbers": numbers
        # print(batch)

        padded_numbers = []
        for n in numbers:
            if (max_length - len(n)) > 0:
                n = n + [1.0] * (max_length - len(n))
            padded_numbers.append(n)
        batch["numbers"] = torch.tensor(padded_numbers, dtype=torch.float)

        padded_labels = []
        for l in labels:
            if (max_length - len(l)) > 0:
                l = l + [self.tokenizer.pad_token_id] * (max_length - len(l))
            padded_labels.append(l)
        # print(padded_labels)
        batch["labels"] = torch.tensor(padded_labels, dtype=torch.long)


        return batch

# custom data collator to include the new numbers parameter in the evaluation process
class CustomDataCollatorForEvaluation:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.default_collator = DataCollatorWithPadding(self.tokenizer, padding='max_length') #DataCollatorWithPadding

    def __call__(self, features):
        # print(features)
        input_ids = [feature["input_ids"] for feature in features]
        numbers = [feature["numbers"] for feature in features]

        max_length_input = len(max(input_ids, key=len))
        max_length_number = len(max(numbers, key=len))

        max_length = max([max_length_input, max_length_number])

        self.default_collator.max_length = max_length
        batch = self.default_collator({"input_ids": input_ids})

        padded_numbers = []
        for n in numbers:
            if (max_length - len(n)) > 0:
                n = n + [1.0] * (max_length - len(n))
            padded_numbers.append(n)
        batch["numbers"] = torch.tensor(padded_numbers, dtype=torch.float)

        return batch


# function to create a synthetic math dataset with the basic arithmetic operators
def create_math_tasks(amount, upper_bound = 1000, lower_bound = -1000):
    tasks = ['+', '-', '*', '/']
    data = {'text': [],
            'output': []}

    for x in range(amount):
        number1 = random.randint(lower_bound, upper_bound)
        number2 = random.randint(lower_bound, upper_bound)

        task = random.choice(tasks)

        if task == '/':
            while number2 == 0:
                number2 = random.randint(lower_bound, upper_bound)

        task_text = f"{number1} {task} {number2}"
        result = eval(task_text)

        if task == '/':
            result = round(result, 4)

        data['text'].append(task_text)
        data['output'].append(str(result))
    ds = Dataset.from_dict(data)
    return ds

# function to create a synthetic math list dataset with the operators: mean, median, min and max
def create_math_lists_tasks(amount, upper_bound = 1000, lower_bound = -1000, max_entries = 10):
    tasks = ['mean', 'median', 'min', 'max']
    data = {'text': [],
            'output': []}

    for x in range(amount):
        numbers = []
        entries = random.randint(1, max_entries)
        task = random.choice(tasks)

        for entry in range(entries):
            numbers.append(random.randint(lower_bound, upper_bound))

        if task == 'mean':
            result = round(np.mean(numbers), 4)
            task_text = f"Determine the mean value of this list: {numbers}"
        if task == 'median':
            result = np.median(numbers)
            task_text = f"Determine the median value of this list: {numbers}"
        if task == 'min':
            result = np.min(numbers)
            task_text = f"Determine the min value of this list: {numbers}"
        if task == 'max':
            result = np.max(numbers)
            task_text = f"Determine the max value of this list: {numbers}"

        data['text'].append(task_text)
        data['output'].append(str(result))
    ds = Dataset.from_dict(data)
    return ds


def plot_lerningcurve(file_path, skip_steps=0, show_learning_rate=True, tick_step_size=0):
    log_file = os.path.join(file_path, "trainer_state.json")
    with open(log_file) as f:
        logs = json.load(f)

    steps = []
    losses = []
    eval_steps = []
    eval_losses = []
    lr = []
    lr_steps = []
    for l in logs['log_history']:
        if 'loss' in l.keys():
            steps.append(l['step'])
            losses.append(l['loss'])
        elif 'train_loss' in l.keys():
            steps.append(l['step'])
            losses.append(l['train_loss'])
        elif 'eval_loss' in l.keys():
            eval_steps.append(l['step'])
            eval_losses.append(l['eval_loss'])
        if 'learning_rate' in l.keys():
            lr_steps.append(l['step'])
            lr.append(l['learning_rate'])

    skip_eval_steps = 0
    fig, ax = plt.subplots(figsize = (15, 10))
    if(skip_steps > 0):
        skip_eval_steps = 1
    ax.plot(steps[skip_steps:], losses[skip_steps:], label='train_loss', linestyle='-')
    ax.plot(eval_steps[skip_eval_steps:], eval_losses[skip_eval_steps:], label='eval_loss', linestyle='--')
    if(tick_step_size > 0):
        ax.set_xticks(range(0, steps[-1], tick_step_size))
    ax.set_ylabel('Loss')
    ax.set_xlabel('Steps')
    ax.set_title('Lernkurve')

    if show_learning_rate:
        ax2 = ax.twinx()
        ax2.plot(lr_steps, lr, label='learning_rate', linestyle='-', color='black')
        ax2.set_ylabel('Learning Rate')

    fig.legend()

    plt.grid(True)

    fig.savefig(os.path.join(file_path, "lernkurve.png"))
    plt.show()


# function to train a model and save the training args
def train_loop(sftconfig, peftconfig, model, tokenizer, train_ds, eval_ds, resume_from_checkpoint = False):


    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=peftconfig,
        tokenizer=tokenizer,
        args=sftconfig,
        data_collator=CustomDataCollator(tokenizer),
    )

    with open(os.path.join(sftconfig.output_dir, "args.json"), 'w') as file:
        json.dump(sftconfig.to_dict(), file)

    peftconfig.save_pretrained(sftconfig.output_dir)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    trainer.save_model(os.path.join(sftconfig.output_dir, "final_save"))
    trainer.save_state()

    plot_lerningcurve(sftconfig.output_dir)




def load_and_merge_custom_model_and_tokenizer(final_save_path, device, new_path=None):

    new_model = AutoPeftModelForCausalLM.from_pretrained(
        final_save_path,
        # low_cpu_mem_usage=True,
        # return_dict=True,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map=device,
        attn_implementation="flash_attention_2",
    )

    finetuned_model = new_model.merge_and_unload()
    finetuned_tokenizer = AutoTokenizer.from_pretrained(final_save_path)
    finetuned_tokenizer.padding_side = 'left'
    if new_path is not None:
        finetuned_model.save_pretrained(new_path)
        finetuned_tokenizer.save_pretrained(new_path)

    del new_model
    # del finetuned_tokenizer
    # del finetuned_model
    torch.cuda.empty_cache()
    return finetuned_model, finetuned_tokenizer


def custom_model_evaluation(data, model, tokenizer, data_collator, batch_size = 2, max_new_tokens = 128):
    model.eval()
    outputs = []
    labels = data["labels"]
    data.remove_columns(["labels"])

    with torch.no_grad():
        for i in tqdm(range(0, len(data), batch_size), desc="Generating responses"):
            torch.cuda.empty_cache()
            # print(data)
            batch_end = i+batch_size
            if batch_end > len(data):
                batch_end = len(data)
            batch = data.select(range(i, batch_end))
            # batch = data[i:i+batch_size]
            # print(batch)

            batch = data_collator(batch)
            input_ids = batch["input_ids"].to(model.device)
            numbers = batch["numbers"].to(model.device)
            # print(input_ids.shape)
            # print(numbers.shape)

            generated_ids = model.generate(
                input_ids=input_ids,
                numbers=numbers,
                max_new_tokens=max_new_tokens,
            )

            for generated_id in generated_ids:
                text = tokenizer.decode(generated_id, skip_special_tokens=True)
                outputs.append(text)

        correct = 0
        total = len(data)
        i = 0

        for entry in tqdm(data, desc="Processing outputs"):
            correct_solution = entry['labels']
            if correct_solution in outputs[i].strip():
                correct += 1
            i += 1

        accuracy = correct / total

    return outputs, accuracy




































