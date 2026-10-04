from __future__ import annotations

import json
import os

import bitsandbytes as bnb
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import Trainer, TrainerCallback, TrainingArguments

from .physics_loss import physics_loss_from_logits, NUM_LABELS


def physics_collate_fn(examples):
    """Like C7's `collate_fn`, but also stacks `raw_params` into a (B, 4) tensor."""

    for example in examples:
        example["input_ids"] = torch.as_tensor(example["input_ids_text"])
        example["attention_mask"] = torch.as_tensor(example["attention_mask_text"])
        example["label"] = torch.as_tensor(example["label"])

    input_ids = torch.stack([example["input_ids"] for example in examples])
    attention_masks = torch.stack([example["attention_mask"] for example in examples])
    input_ids = torch.squeeze(input_ids, dim=1)
    attention_masks = torch.squeeze(attention_masks, dim=1)
    labels = torch.stack([example["label"] for example in examples])

    raw_params = torch.tensor(
        [example["raw_params"] for example in examples],
        dtype=torch.float32,
    )

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels,
        "raw_params": raw_params,
    }


class _LossComponentLogger(TrainerCallback):
    """Light-weight callback that flushes per-step CE / physics-loss
    components to a JSONL file. The standard `Trainer.log` only tracks
    the total loss, so we add this so §5.6 Part B can plot the
    components vs steps.
    """

    def __init__(self, out_path: str):
        self.out_path = out_path
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                f.write("")
        except OSError:
            pass

    def log_step(self, step: int, ce: float, phys: float, n_valid: int, lam: float):
        try:
            with open(self.out_path, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "step": step,
                            "ce_loss": ce,
                            "physics_loss": phys,
                            "n_valid_phys_rows": n_valid,
                            "lambda_phys": lam,
                        }
                    )
                    + "\n"
                )
        except OSError:
            pass


class PhysicsConstrainedSFTTrainer(Trainer):
    """Trainer with composite loss = CE + λ · physics-violation."""

    def __init__(self, lambda_phys: float = 1.0, num_labels: int = NUM_LABELS, loss_log_path: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.lambda_phys = float(lambda_phys)
        self.num_labels = int(num_labels)
        self._loss_logger = _LossComponentLogger(loss_log_path) if loss_log_path else None
        self._step_seen = 0

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        raw_params = inputs.pop("raw_params", None)

        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if hasattr(outputs, "loss") and outputs.loss is not None:
            ce_loss = outputs.loss
        else:
            ce_loss = F.cross_entropy(logits, labels)

        if raw_params is None:
            total_loss = ce_loss
            phys_val = 0.0
            n_valid = 0
        else:
            raw_params = raw_params.to(logits.device).to(torch.float32)
            phys_loss, valid_count = physics_loss_from_logits(
                logits=logits.float(),
                raw_params=raw_params,
                num_labels=self.num_labels,
            )
            total_loss = ce_loss + self.lambda_phys * phys_loss
            phys_val = float(phys_loss.detach().item())
            n_valid = int(valid_count.detach().item())

        self._step_seen += 1
        if self._loss_logger is not None and (
            getattr(self, "is_world_process_zero", lambda: True)()
        ):
            self._loss_logger.log_step(
                step=self._step_seen,
                ce=float(ce_loss.detach().item()),
                phys=phys_val,
                n_valid=n_valid,
                lam=self.lambda_phys,
            )

        return (total_loss, outputs) if return_outputs else total_loss


class ModelFinetunerPhysics:
    """Drop-in replacement for `ModelFinetuner` that uses the physics
    trainer. Public API is identical except for two extra kwargs
    (`lambda_phys`, `loss_log_path`).
    """

    def __init__(self) -> None:
        pass

    def print_trainable_parameters(self, model, use_4bit: bool = False):
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
            f"All Parameters: {all_param:,d} || "
            f"Trainable Parameters: {trainable_params:,d} || "
            f"Trainable Parameters %: {100 * trainable_params / all_param}"
        )

    def find_all_linear_names(self, model):
        cls = bnb.nn.Linear4bit
        lora_module_names = set()
        for name, module in model.named_modules():
            if isinstance(module, cls):
                names = name.split(".")
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])
        if "lm_head" in lora_module_names:
            lora_module_names.remove("lm_head")
        print(f"LoRA module names: {list(lora_module_names)}")
        return list(lora_module_names)

    def fine_tune(
        self,
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
        target_modules: str = "all-linear",
        learning_rate: float = 2e-4,
        flag_fine_tuning: bool = True,
        lambda_phys: float = 1.0,
    ):
        if not flag_fine_tuning:
            for param in model.parameters():
                param.requires_grad = False
            for param in model.score.parameters():
                param.requires_grad = True

        print(f"fine-tuning C10 (physics, lambda_phys={lambda_phys})")

        model.gradient_checkpointing_enable()
        model = prepare_model_for_kbit_training(model)

        target_modules = self.find_all_linear_names(model)

        modules_to_save = ["score"]
        peft_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias=bias,
            task_type=task_type,
            modules_to_save=modules_to_save,
            exclude_modules=modules_to_save,
        )

        model = get_peft_model(model, peft_config)

        self.print_trainable_parameters(model)

        args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=train_epochs,
            per_device_train_batch_size=per_device_train_batch_size,
            per_device_eval_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=1,
            learning_rate=learning_rate,
            logging_steps=10,
            fp16=True,
            weight_decay=0.001,
            max_grad_norm=0.3,
            max_steps=-1,
            warmup_ratio=0.03,
            lr_scheduler_type="cosine",
            report_to="none",
            save_strategy="epoch",
            gradient_checkpointing=True,
            optim="paged_adamw_32bit",
            remove_unused_columns=False,
            ddp_find_unused_parameters=False,
        )

        loss_log_path = os.path.join(str(output_dir), "loss_components.jsonl")

        trainer = PhysicsConstrainedSFTTrainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            data_collator=physics_collate_fn,
            processing_class=tokenizer,
            lambda_phys=lambda_phys,
            num_labels=NUM_LABELS,
            loss_log_path=loss_log_path,
        )

        model.config.use_cache = False

        print("Training C10 ...")

        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_model()
        tokenizer.save_pretrained(output_dir)

        print(metrics)

        try:
            with open(os.path.join(str(output_dir), "physics_meta.json"), "w", encoding="utf-8") as f:
                json.dump({"lambda_phys": float(lambda_phys)}, f)
        except OSError:
            pass

        del model
        del trainer
        torch.cuda.empty_cache()


def main():
    _ = ModelFinetunerPhysics()


if __name__ == "__main__":
    main()
