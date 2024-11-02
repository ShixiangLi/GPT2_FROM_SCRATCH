import torch
import tiktoken
import argparse
import logging
import sys
import os
import time
import json
import tqdm
import psutil
from datetime import datetime
from functools import partial

import torch.ao.quantization

from gpt2 import GPTModel
from dataset import *
from utils import *
from config import CONFIG
from gpt_download import download_and_load_gpt2

parser = argparse.ArgumentParser("gpt2_for_classifier")
parser.add_argument('--train_from_scratch', type=bool, default=0, help='train from scratch')
parser.add_argument('--task', type=str, default='ins', help='task')
parser.add_argument('--model', type=str, default='GPT_CONFIG_355M', help='model name')
parser.add_argument('--num_class', type=int, default=2, help='number of classes')
parser.add_argument('--model_size', type=str, default='355M', help='model size')
parser.add_argument('--model_dir', type=str, default='gpt2', help='model directory')
parser.add_argument('--data', type=str, default='./dataset/instruction-data.json', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.00005, help='init learning rate')
parser.add_argument('--weight_decay', type=float, default=0.1, help='weight decay')
parser.add_argument('--report_freq', type=float, default=5, help='report frequency')
parser.add_argument('--num_epochs', type=int, default=2, help='num of training epochs')
parser.add_argument('--num_workers', type=int, default=0, help='number of workers to generate data')
parser.add_argument('--model_path', type=str, default='./checkpoints/', help='path to save the model')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--llm_to_evaluate', type=str, default='qwen2.5-7b', help='llm')

args = parser.parse_args()

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"EP {epoch + 1} (Step {global_step:06d}):"
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}")
        
        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen


def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()
    input_ids = tokenizer.encode(text)         
    supported_context_length = model.pos_emb.weight.shape[1]
    input_ids = input_ids[:min(max_length, supported_context_length)]

    input_ids += [pad_token_id] * (max_length - len(input_ids))

    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
    predicted_label = torch.argmax(logits, dim=-1).item()

    return "spam" if predicted_label == 1 else "not spam"

def main():

    # ==================================
    # === Set up the experiment=========
    # ==================================

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ==================================
    # === Load the model ===============
    # ==================================
   
    config = CONFIG[args.model]
    model = GPTModel(config)
    tokenizer = tiktoken.get_encoding("gpt2")

    if args.train_from_scratch:
        pass
    else:
        models_dir = args.model_dir
        settings, params = download_and_load_gpt2(
            model_size=args.model_size, models_dir=models_dir
        )
        load_weights_into_gpt(model, params)

        

    
    model = model.to(device)

    # ==================================
    # === Load the data ================
    # ==================================

    file_path = args.data
    url = (
        "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch"
        "/main/ch07/01_main-chapter-code/instruction-data.json"
    )
    data = download_and_load_file(file_path, url)
    print("Number of entries:", len(data))

    train_portion = int(len(data) * 0.85)   
    test_portion = int(len(data) * 0.1)           
    val_portion = len(data) - train_portion - test_portion

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]
    
    customized_collate_fn = partial(
        custom_collate_fn,
        device=device,
        allowed_max_length=1024
    )
    
    # num_workers = 0     
    # batch_size = 8

    torch.manual_seed(123)

    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers
    )
    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers
    )
    test_dataset = InstructionDataset(test_data, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers
    )

    # ==================================
    # === Train the model ==============
    # ==================================

    start_time = time.time()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=args.num_epochs, eval_freq=5, eval_iter=5,
        start_context=format_input(val_data[0]), tokenizer=tokenizer
    )
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    # ==================================
    # === Evaluate the model ===========
    # ==================================

    # Evaluate the model on the test set
    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
        input_text = format_input(entry)

        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
                max_new_tokens=256,
                context_size=config["context_length"],
                eos_id=50256
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)

        response_text = (
            generated_text[len(input_text):]
            .replace("### Response:", "")
            .strip()
        )
        test_data[i]["model_response"] = response_text

    with open("./instruct_finetune_test/instruction-data-with-response.json", "w") as file:
        json.dump(test_data, file, indent=4)

    # check if ollama is running
    ollama_running = check_if_running("ollama")
    if not ollama_running:
        raise RuntimeError(
            "Ollama not running. Launch ollama before proceeding."
        )
    print("Ollama running:", check_if_running("ollama"))

    # Evaluate the model using ollama
    for entry in test_data[:3]:
        prompt = (
            f"Given the input `{format_input(entry)}` "
            f"and correct output `{entry['output']}`, "
            f"score the model response `{entry['model_response']}`"
            f" on a scale from 0 to 100, where 100 is the best score. "
        )
        print("\nDataset response:")
        print(">>", entry['output'])
        print("\nModel response:")
        print(">>", entry["model_response"])
        print("\nScore:")
        print(">>", query_model(prompt, model=args.llm_to_evaluate))
        print("\n-------------------------")

    # Generate model scores
    scores = generate_model_scores(test_data, "model_response", model=args.llm_to_evaluate)
    print(f"Number of scores: {len(scores)} of {len(test_data)}")
    print(f"Average score: {sum(scores)/len(scores):.2f}\n")

    # ==================================
    # === Save the model ===============
    # ==================================

    file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + '_' + args.model + ".pth"
    task_name = args.task
    save_path = args.model_path + task_name + '_' + file_name
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        }, 
        save_path
    )
    print(f"Model saved as {file_name}")

    # ==================================
    # === Test the model ===============
    # ==================================

    for entry in test_data[:3]:     
        input_text = format_input(entry)
        token_ids = generate(              
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=config["context_length"],
            eos_id=50256
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        response_text = (
            generated_text[len(input_text):]
            .replace("### Response:", "")
            .strip()
        )
        print(input_text)
        print(f"\nCorrect response:\n>> {entry['output']}")
        print(f"\nModel response:\n>> {response_text.strip()}")
        print("-------------------------------------")

if __name__ == "__main__":
    main()
