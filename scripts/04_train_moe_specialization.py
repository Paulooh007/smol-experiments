import torch
import itertools
import time
from torch.optim import AdamW
from torch.amp import autocast
from transformers import AutoTokenizer

from src.config import SmolMoEConfig
from src.modeling_moe import SmolMoELM
from src.data import get_specialized_dataset, specialized_collate_fn
from src.trainers import compute_router_guidance_loss, causal_lm_loss

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = SmolMoEConfig()
    
    # Load upcycled weights (Ensure script 03 has been run or weights downloaded)
    # For this example, we assume weights are at 'models/upcycled_moe.pt'
    try:
        model = SmolMoELM(config).to(device).to(torch.bfloat16)
        # In a real scenario, load the state dict here:
        # model.load_state_dict(torch.load("models/upcycled_moe.pt"))
        print("Model initialized (Weights should be loaded here)")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    
    # Prepare Data
    loader = torch.utils.data.DataLoader(
        get_specialized_dataset(tokenizer),
        batch_size=4,
        collate_fn=specialized_collate_fn
    )
    
    # Hyperparameters
    ROUTER_STEPS = 600
    JOINT_STEPS = 400
    TOTAL_STEPS = ROUTER_STEPS + JOINT_STEPS
    
    print("Starting Specialization Training...")
    
    # Optimizer setup (Stage A: Router only)
    router_params = [p for n, p in model.named_parameters() if "moe.gate" in n]
    optimizer = AdamW(router_params, lr=3e-4)
    
    train_iter = itertools.cycle(loader)
    start_time = time.time()
    
    for step in range(1, TOTAL_STEPS + 1):
        # Switch to Stage B (Joint)
        if step == ROUTER_STEPS + 1:
            print(">>> Switching to Stage B: Joint Tuning")
            optimizer = AdamW(model.parameters(), lr=3e-5)
            
        batch = next(train_iter)
        inputs = batch["input_ids"].to(device)
        domains = batch["domain_ids"].to(device)
        
        with autocast("cuda", dtype=torch.bfloat16):
            outputs = model(inputs, attention_mask=batch["attention_mask"].to(device))
            
            lm_loss = causal_lm_loss(outputs['logits'], inputs)
            guidance_loss = compute_router_guidance_loss(outputs['router_logits'], domains, config)
            
            # Heavy weight on guidance during Stage A
            guidance_weight = 20.0 if step <= ROUTER_STEPS else 5.0
            loss = lm_loss + (guidance_weight * guidance_loss)
            
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if step % 50 == 0:
            print(f"Step {step}: Loss={loss.item():.3f} (Guidance={guidance_loss.item():.4f})")

    print("Training Complete.")
    # torch.save(model.state_dict(), "models/specialized_moe.pt")

if __name__ == "__main__":
    main()