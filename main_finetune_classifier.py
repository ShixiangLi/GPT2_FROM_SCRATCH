import torch
import tiktoken
import argparse
import logging
import sys
import os
import time
from datetime import datetime

import torch.ao.quantization

from gpt2 import GPTModel
from dataset import *
from utils import *
from config import CONFIG
from gpt_download import download_and_load_gpt2

parser = argparse.ArgumentParser("gpt2_for_classifier")
parser.add_argument('--train_from_scratch', type=bool, default=1, help='train from scratch')
parser.add_argument('--task', type=str, default='cls', help='task')
parser.add_argument('--model', type=str, default='GPT_CONFIG_124M', help='model name')
parser.add_argument('--num_class', type=int, default=2, help='number of classes')
parser.add_argument('--model_size', type=str, default='124M', help='model size')
parser.add_argument('--model_dir', type=str, default='gpt2', help='model directory')
parser.add_argument('--data', type=str, default='./dataset/sms_spam_collection/', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=5e-5, help='init learning rate')
parser.add_argument('--weight_decay', type=float, default=0.1, help='weight decay')
parser.add_argument('--report_freq', type=float, default=5, help='report frequency')
parser.add_argument('--num_epochs', type=int, default=5, help='num of training epochs')
parser.add_argument('--model_path', type=str, default='./checkpoints/', help='path to save the model')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--seed', type=int, default=123, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
args = parser.parse_args()

def train_classifier_simple(model, train_loader, val_loader, optimizer, device,
                            num_epochs, eval_freq, eval_iter):
    train_losses, val_losses, train_accs, val_accs = [], [], [], []  
    examples_seen, global_step = 0, -1
    for epoch in range(num_epochs):   
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()                     
            loss = calc_loss_batch(input_batch, target_batch, model, device, task=args.task)
            loss.backward()                         
            optimizer.step()                         
            examples_seen += input_batch.shape[0]   
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter, task=args.task)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(
                    f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}"
                )
        
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)

        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
    
    return train_losses, val_losses, train_accs, val_accs, examples_seen

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

        # Freeze the model
        for param in model.parameters():
            param.requires_grad = False

        # Add a classifier head
        num_classes = args.num_class
        model.out_head = torch.nn.Linear(
            in_features=config["emb_dim"], 
            out_features=num_classes
        )

        # Enable the output head
        for param in model.trf_blocks[-1].parameters():
            param.requires_grad = True
        for param in model.final_norm.parameters():
            param.requires_grad = True
        

    
    model = model.to(device)

    # ==================================
    # === Load the data ================
    # ==================================

    file_path = args.data

    train_dataset = SpamDataset(
        csv_file=file_path + "train.csv",
        max_length=None,
        tokenizer=tokenizer
    )
    val_dataset = SpamDataset(
        csv_file=file_path + "validation.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
    )
    test_dataset = SpamDataset(
        csv_file=file_path + "test.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
    )

    num_workers = 0
    batch_size = 8
    torch.manual_seed(123)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )


    # ==================================
    # === Train the model ==============
    # ==================================

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    start_time = time.time()
    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=args.num_epochs, eval_freq=50,
        eval_iter=5
    )
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    train_accuracy = calc_accuracy_loader(train_loader, model, device)
    val_accuracy = calc_accuracy_loader(val_loader, model, device)
    test_accuracy = calc_accuracy_loader(test_loader, model, device)
    print(f"Training accuracy: {train_accuracy*100:.2f}%")
    print(f"Validation accuracy: {val_accuracy*100:.2f}%")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")

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

    # ==================================
    # === Evaluate the model ===========
    # ==================================

    text_1 = "You have won a free ticket to the Bahamas! Text 'WIN' to 12345 to claim your prize!"
    print(classify_review(text_1, model, tokenizer, device, max_length=train_dataset.max_length))

    text_2 = "Hey, how are you doing? Do you want to meet up later?"
    print(classify_review(text_2, model, tokenizer, device, max_length=train_dataset.max_length))   

if __name__ == "__main__":
    main()
