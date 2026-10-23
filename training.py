import os
import sys
import json
import argparse
import random
import torch
import torch.optim as optim
from model import MeyTokenizer, MeyTransformer

def get_batch(data, block_size, batch_size, device='cpu'):
    # Ensure data is long enough
    if len(data) <= block_size:
        times = (block_size + 1) // max(1, len(data)) + 1
        data = data.repeat(times)
    max_start = len(data) - block_size
    ix = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, eval_iters=30, device='cpu'):
    out = {}
    model.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, block_size, batch_size, device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out

def update_status(status_file, data):
    try:
        with open(status_file, 'w') as f:
            json.dump(data, f)
    except Exception as e:
        print(f"Error updating status file: {e}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description="Mey AI Training Script")
    parser.add_argument('--text_path', type=str, required=True, help="Path to text corpus")
    parser.add_argument('--model_path', type=str, required=True, help="Path to save PyTorch model weights")
    parser.add_argument('--vocab_path', type=str, required=True, help="Path to save vocab json")
    parser.add_argument('--status_path', type=str, required=True, help="Path to save status json")
    parser.add_argument('--iters', type=int, default=1000, help="Number of training iterations")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
    parser.add_argument('--block_size', type=int, default=64, help="Context length")
    parser.add_argument('--embed_dim', type=int, default=128, help="Embedding dimension")
    parser.add_argument('--num_heads', type=int, default=4, help="Attention heads")
    parser.add_argument('--num_layers', type=int, default=4, help="Transformer layers")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--weight_decay', type=float, default=0.01, help="AdamW weight decay")
    parser.add_argument('--patience', type=int, default=8, help="Evaluations without improvement before early stopping")
    parser.add_argument('--eval_iters', type=int, default=20, help="Number of batches used for each evaluation")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducible training")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Set status to starting
    status = {
        "status": "starting",
        "step": 0,
        "total_steps": args.iters,
        "loss": 0.0,
        "val_loss": 0.0,
        "message": "Initializing data...",
        "history": []
    }
    update_status(args.status_path, status)

    if not os.path.exists(args.text_path):
        status["status"] = "error"
        status["message"] = f"Dataset file not found: {args.text_path}"
        update_status(args.status_path, status)
        sys.exit(1)

    try:
        # Load and tokenize dataset
        with open(args.text_path, 'r', encoding='utf-8') as f:
            text = f.read()

        tokenizer = MeyTokenizer()
        tokenizer.fit(text)
        tokenizer.save(args.vocab_path)

        data_list = tokenizer.encode(text)
        
        # Ensure we have enough data
        if len(data_list) <= args.block_size:
            times = (args.block_size + 1) // max(1, len(data_list)) + 1
            data_list = (data_list * times)[:(args.block_size + 1) * 2]

        data = torch.tensor(data_list, dtype=torch.long)
        n = int(0.9 * len(data))
        train_data = data[:n]
        val_data = data[n:]

        # Initialize model
        device = 'cpu' # Force CPU since it's local Alder Lake without NVIDIA
        model = MeyTransformer(
            vocab_size=tokenizer.vocab_size,
            embed_dim=args.embed_dim,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            block_size=args.block_size,
            dropout=0.1
        ).to(device)

        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.iters))

        status["status"] = "training"
        status["message"] = f"Model parameters: {sum(p.numel() for p in model.parameters()):,}"
        update_status(args.status_path, status)

        eval_interval = max(10, args.iters // 20)
        history = []
        best_val_loss = float('inf')
        best_state = None
        evaluations_without_improvement = 0

        for step in range(args.iters + 1):
            xb, yb = get_batch(train_data, args.block_size, args.batch_size, device)
            _, loss = model(xb, yb)
            
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if step < args.iters:
                scheduler.step()

            # Periodic evaluation
            if step % eval_interval == 0 or step == args.iters:
                losses = estimate_loss(
                    model,
                    train_data,
                    val_data,
                    args.block_size,
                    args.batch_size,
                    eval_iters=args.eval_iters,
                    device=device,
                )
                
                history.append({
                    "step": step,
                    "train_loss": round(losses['train'], 4),
                    "val_loss": round(losses['val'], 4)
                })

                status["step"] = step
                status["loss"] = round(loss.item(), 4)
                status["val_loss"] = round(losses['val'], 4)
                status["message"] = f"Iteracja {step}/{args.iters} - Loss: {loss.item():.4f}"
                status["history"] = history
                update_status(args.status_path, status)
                print(f"Step {step}/{args.iters}: Train Loss {losses['train']:.4f}, Val Loss {losses['val']:.4f}")

                if losses['val'] < best_val_loss:
                    best_val_loss = losses['val']
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    evaluations_without_improvement = 0
                else:
                    evaluations_without_improvement += 1

                if evaluations_without_improvement >= args.patience:
                    status["status"] = "stopping"
                    status["message"] = "Wczesne zatrzymanie: walidacja przestała się poprawiać."
                    update_status(args.status_path, status)
                    print("Early stopping: validation loss stopped improving.")
                    break
            else:
                # Update progress every step, but don't re-calculate validation loss
                if step % 5 == 0:
                    status["step"] = step
                    status["loss"] = round(loss.item(), 4)
                    status["message"] = f"Iteracja {step}/{args.iters} - Loss: {loss.item():.4f}"
                    update_status(args.status_path, status)

        # Save model checkpoint
        if best_state is not None:
            model.load_state_dict(best_state)

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'vocab_size': tokenizer.vocab_size,
            'hyperparameters': {
                'embed_dim': args.embed_dim,
                'num_heads': args.num_heads,
                'num_layers': args.num_layers,
                'block_size': args.block_size
            },
            'best_val_loss': best_val_loss,
            'training_args': vars(args),
        }
        torch.save(checkpoint, args.model_path)

        # Complete status
        status["status"] = "completed"
        status["step"] = args.iters
        status["message"] = "Trening zakończony sukcesem!"
        update_status(args.status_path, status)
        print("Training completed and saved.")

    except Exception as e:
        status["status"] = "error"
        status["message"] = f"Training failed: {str(e)}"
        update_status(args.status_path, status)
        print(f"Error during training: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
