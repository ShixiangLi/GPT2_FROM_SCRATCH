import torch
import tiktoken
import argparse
import logging
import sys
import os
from datetime import datetime

import torch.ao.quantization

from gpt2 import GPTModel
from dataset import *
from utils import *
from config import CONFIG
from gpt_download import download_and_load_gpt2

parser = argparse.ArgumentParser("gpt2_from")
parser.add_argument('--train_from_scratch', type=bool, default=1, help='train from scratch')
parser.add_argument('--model', type=str, default='GPT_CONFIG_124M', help='model name')
parser.add_argument('--model_size', type=str, default='124M', help='model size')
parser.add_argument('--model_dir', type=str, default='gpt2', help='model directory')
parser.add_argument('--data', type=str, default='./dataset/the-verdict.txt', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.0004, help='init learning rate')
parser.add_argument('--weight_decay', type=float, default=0.1, help='weight decay')
parser.add_argument('--report_freq', type=float, default=5, help='report frequency')
parser.add_argument('--num_epochs', type=int, default=10, help='num of training epochs')
parser.add_argument('--model_path', type=str, default='./checkpoints/', help='path to save the model')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
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
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()

    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    train_loader = create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=config["context_length"],
        stride=config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
        )
    val_loader = create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=config["context_length"],
        stride=config["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
        )

    if args.train_from_scratch:
        # ==================================
        # === Train the model ==============
        # ==================================

        optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
        train_losses, val_losses, tokens_seen = train_model_simple(
            model, train_loader, val_loader, optimizer, device, 
            num_epochs=args.num_epochs, eval_freq=args.report_freq, eval_iter=5, 
            start_context="Every effort moves you", tokenizer=tokenizer
        )

        # ==================================
        # === Save the model ===============
        # ==================================

        file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_") + args.model + ".pth"
        save_path = args.model_path + file_name
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            }, 
            save_path
        )
    else:
        token_ids = generate(
            model=model,
            idx=text_to_token_ids("I am so upset, but", tokenizer).to(device),
            max_new_tokens=30,
            context_size=config["context_length"],
            top_k=50,
            temperature=1.5
        )
        print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

if __name__ == "__main__":
    main()
