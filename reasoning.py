"""Local reasoning loop built from Mey's own PyTorch checkpoint."""

import torch

from model import MeyTokenizer, MeyTransformer


class MeyReasoner:
    def __init__(self, model_path, vocab_path):
        self.tokenizer = MeyTokenizer()
        self.tokenizer.load(vocab_path)
        checkpoint = torch.load(model_path, map_location="cpu")
        hp = checkpoint.get("hyperparameters", {})
        self.model = MeyTransformer(
            vocab_size=checkpoint["vocab_size"],
            embed_dim=hp.get("embed_dim", 128),
            num_heads=hp.get("num_heads", 4),
            num_layers=hp.get("num_layers", 4),
            block_size=hp.get("block_size", 64),
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    @torch.no_grad()
    def _generate(self, prompt, max_new_tokens=48):
        token_ids = self.tokenizer.encode(prompt)
        if not token_ids:
            return ""
        token_ids = token_ids[-self.model.block_size + 1:]
        idx = torch.tensor([token_ids], dtype=torch.long)
        output = self.model.generate(idx, max_new_tokens=max_new_tokens, temperature=0.55, top_k=12)
        return self.tokenizer.decode(output[0][len(token_ids):].tolist())

    def answer(self, question, context=""):
        plan_prompt = (
            "Zadanie: przeanalizuj pytanie. Wypisz krótki plan rozwiązania, "
            "ważne założenia i sprawdzenie błędów. Nie odpowiadaj jeszcze.\n"
            f"Pytanie: {question}"
        )
        plan = self._generate(plan_prompt)
        answer_prompt = (
            "Jesteś Mey. Użyj planu do udzielenia krótkiej, poprawnej odpowiedzi. "
            "Nie pokazuj planu ani toku rozumowania.\n"
            f"Plan wewnętrzny: {plan}\n"
            f"Kontekst: {context}\n"
            f"Pytanie: {question}\nOdpowiedź:"
        )
        return self._generate(answer_prompt, max_new_tokens=72).strip()

