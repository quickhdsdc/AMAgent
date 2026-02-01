import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import bitsandbytes as bnb
from trl import SFTTrainer
from transformers import TrainingArguments
from accelerate import Accelerator


def collate_fn(examples):
    for example in examples:
        example['input_ids'] = torch.as_tensor(example['input_ids_text'])
        example['attention_mask'] = torch.as_tensor(example['attention_mask_text'])
        example['label'] = torch.as_tensor(example['label'])
       
    input_ids = torch.stack([example["input_ids"] for example in examples])
    attention_masks = torch.stack([example["attention_mask"] for example in examples])
    input_ids = torch.squeeze(input_ids, dim=1)
    attention_masks = torch.squeeze(attention_masks, dim=1)
    labels = torch.stack([example["label"] for example in examples])
    return {
        "input_ids": input_ids, 
        "attention_mask": attention_masks,
        "labels": labels
    }


class ModelFinetuner:
    def __init__(self) -> None:
        pass

    def print_trainable_parameters(self, model, use_4bit = False):
        """Prints the number of trainable parameters in the model."""
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            num_params = param.numel()
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel
            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params

        if use_4bit:
            trainable_params /= 2
        print(
            f"All Parameters: {all_param:,d} || Trainable Parameters: {trainable_params:,d} || Trainable Parameters %: {100 * trainable_params / all_param}"
        )

    def find_all_linear_names(self, model):
        """
        Find modules to apply LoRA to.
        """
        cls = bnb.nn.Linear4bit
        lora_module_names = set()
        
        for name, module in model.named_modules():
            if isinstance(module, cls):
                names = name.split('.')
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])

        if 'lm_head' in lora_module_names:
            lora_module_names.remove('lm_head')
        
        return list(lora_module_names)
    
    def fine_tune(self, 
                  model,
                  tokenizer,
                  train_ds,
                  val_ds,
                  lora_r,
                  lora_alpha,
                  lora_dropout,
                  bias,
                  task_type,
                  per_device_train_batch_size,
                  output_dir,
                  train_epochs,
                  target_modules="all-linear",
                  learning_rate = 2e-4,
                  flag_fine_tuning = True,
                  accelerator: Accelerator = None
                  ):
        if not flag_fine_tuning:
            # Freeze the base model parameters
            for param in model.parameters():
                param.requires_grad = False
            # Make sure the `score` module's parameters are trainable
            for param in model.score.parameters():
                param.requires_grad = True

        if not accelerator or accelerator.is_main_process:
            print('fine-tuning the peft model')
        
        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)
        
        target_modules = self.find_all_linear_names(model)
        
        if not accelerator or accelerator.is_main_process:
            print(target_modules)
        
        modules_to_save = ['score']
            
        peft_config = LoraConfig(
            r = lora_r,
            lora_alpha = lora_alpha,
            target_modules = target_modules,
            lora_dropout = lora_dropout,
            bias = bias,
            task_type = task_type,
            modules_to_save = modules_to_save,
            exclude_modules = modules_to_save
        )

        model = get_peft_model(model, peft_config)
        
        if not accelerator or accelerator.is_main_process:
            self.print_trainable_parameters(model)

        args = TrainingArguments(
                output_dir = output_dir,
                num_train_epochs=train_epochs,
                per_device_train_batch_size = per_device_train_batch_size,
                per_device_eval_batch_size=per_device_train_batch_size,
                gradient_accumulation_steps = 1,
                learning_rate = learning_rate, 
                logging_steps=10,
                fp16 = True,
                weight_decay=0.001,
                max_grad_norm=0.3,
                max_steps=-1,
                warmup_ratio=0.03,
                group_by_length=True,
                lr_scheduler_type="cosine",
                report_to="tensorboard",
                save_strategy="epoch",
                gradient_checkpointing=True,
                optim="paged_adamw_32bit",
                remove_unused_columns=False,
                ddp_find_unused_parameters=False,
        )
        
        data_collator = collate_fn

        trainer = SFTTrainer(
                model=model,
                args=args,
                train_dataset=train_ds,
                peft_config=peft_config,
                dataset_text_field="text",
                tokenizer=tokenizer,
                packing=False,
                max_seq_length=4096,
                data_collator = data_collator,
                dataset_kwargs={
                    "add_special_tokens": False,
                    "append_concat_token": False,
                }
            )

        model.config.use_cache = False
        do_train = True

        if not accelerator or accelerator.is_main_process:
            print("Training...")

        if do_train:
            train_result = trainer.train()
            metrics = train_result.metrics
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_model()
            tokenizer.save_pretrained(output_dir)
            if not accelerator or accelerator.is_main_process:
                print(metrics)

        if not accelerator or accelerator.is_main_process:
            print('Evaluation...')

        del model
        del trainer
        torch.cuda.empty_cache()

def main():
    pass

if __name__ == "__main__":
    main()