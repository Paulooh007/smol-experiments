import torch
import evaluate

def process_in_batches(ds, model, tokenizer, generate_params, batch_size=16, field="src"):
    """Generates text in batches for evaluation."""
    outputs = []
    start = 0
    
    while start < len(ds[field]):
        end = min(start + batch_size, len(ds[field]))
        batch = ds[field][start:end]
        
        # Basic formatting if needed, otherwise pass raw
        prompts = [t + " ###>" for t in batch] if "###>" not in batch[0] else batch

        try:
            inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(model.device)
            with torch.no_grad():
                gen_ids = model.generate(**inputs, **generate_params, pad_token_id=tokenizer.eos_token_id)
            
            decoded = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
            # Extract response after separator
            res = [txt.split("###>")[-1].strip() for txt in decoded]
            outputs.extend(res)
            start = end
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("OOM. Halving batch size.")
                batch_size //= 2
                torch.cuda.empty_cache()
                if batch_size == 0: raise e
            else:
                raise e
                
    return outputs

def compute_bleu(model, tokenizer, dataset):
    """Computes BLEU score on a test dataset."""
    gen_params = {"max_new_tokens": 128, "do_sample": True, "num_beams": 1}
    preds = process_in_batches(dataset, model, tokenizer, gen_params, batch_size=8)
    targets = dataset["tgt"]
    
    bleu = evaluate.load("bleu")
    results = bleu.compute(predictions=preds, references=targets)
    return results["bleu"]