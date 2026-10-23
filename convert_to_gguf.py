import os
import sys
import json
import torch
import numpy as np
from gguf import GGUFWriter

def main():
    pt_path = "/home/eryk/mey_ai/data/mey_custom.pt"
    vocab_path = "/home/eryk/mey_ai/data/mey_vocab.json"
    output_dir = "/home/eryk/Pobrane"
    output_path = os.path.join(output_dir, "mey_custom.gguf")

    if not os.path.exists(pt_path):
        print(f"Błąd: Plik {pt_path} nie istnieje.")
        sys.exit(1)
        
    if not os.path.exists(vocab_path):
        print(f"Błąd: Plik {vocab_path} nie istnieje.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Ładowanie checkpointu PyTorch z {pt_path}...")
    checkpoint = torch.load(pt_path, map_location='cpu')
    
    state_dict = checkpoint['model_state_dict']
    vocab_size = checkpoint['vocab_size']
    hp = checkpoint.get('hyperparameters', {})
    
    print(f"Model zdefiniowany: vocab_size={vocab_size}, hp={hp}")
    
    print(f"Ładowanie słownika z {vocab_path}...")
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    tokens = vocab_data['vocab']
    
    # Initialize GGUFWriter
    print(f"Tworzenie pliku GGUF: {output_path}...")
    writer = GGUFWriter(output_path, "mey_transformer")
    
    # Add metadata
    writer.add_name("Mey 1.0 Custom PyTorch")
    writer.add_uint32("mey.vocab_size", vocab_size)
    writer.add_uint32("mey.context_length", hp.get('block_size', 64))
    writer.add_uint32("mey.embedding_length", hp.get('embed_dim', 128))
    writer.add_uint32("mey.block_count", hp.get('num_layers', 4))
    writer.add_uint32("mey.head_count", hp.get('num_heads', 4))
    
    # Write tokenizer tokens
    writer.add_token_list(tokens)
    
    # Add tensors from state_dict
    for name, tensor in state_dict.items():
        # Convert torch tensor to numpy float32
        arr = tensor.detach().cpu().numpy().astype(np.float32)
        print(f"Dodawanie tensora: {name} z kształtem {arr.shape}...")
        writer.add_tensor(name, arr)
        
    # Write everything
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    
    print(f"\nSukces! Plik GGUF został pomyślnie utworzony pod ścieżką:\n{output_path}")

if __name__ == "__main__":
    main()
