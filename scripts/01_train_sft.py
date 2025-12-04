import os
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from src.data import format_text

def main():
    model_name = "HuggingFaceTB/SmolLM-135M"
    output_dir = "./outputs/sft/"
    
    print(f"Loading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load and filter dataset
    ds = load_dataset("grammarly/coedit", split="train")
    ds = ds.filter(lambda x: x['task'] == 'gec')
    
    # Apply format
    ds = ds.map(format_text, remove_columns=ds.column_names)
    split_ds = ds.train_test_split(test_size=0.1)
    
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    
    print("Starting SFT Training...")
    sft_config = SFTConfig(
        output_dir=output_dir,
        packing=True,
        dataset_batch_size=12,
        learning_rate=7e-5,
        num_train_epochs=3,
        logging_steps=50,
        save_strategy="epoch"
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=split_ds["train"],
        eval_dataset=split_ds["test"],
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=sft_config,
        max_seq_length=512
    )
    
    trainer.train()
    trainer.save_model(output_dir + "final")
    print(f"SFT Model saved to {output_dir}final")

if __name__ == "__main__":
    main()