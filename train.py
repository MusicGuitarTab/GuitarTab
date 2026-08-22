#!/usr/bin/env python3
"""
Training script for Fretting-Transformer using Hydra configuration.
"""

from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, random_split
from tqdm import tqdm
import hydra
from transformers.modeling_outputs import Seq2SeqLMOutput
from omegaconf import DictConfig, OmegaConf

from src.tab_dataset import TabDataset
from src.model import FrettingTransformer
from src.metrics import TabAccuracyMetrics, generate_and_compute_accuracy
from src.dataloader import create_dataset, create_dataloader
from src.training_logger import TrainingLogger, save_generated_samples


from src.tab_dataset import TabDatasetBatchInput


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_dataloaders(cfg: DictConfig) -> tuple[DataLoader, DataLoader, DataLoader, TabDataset]:
    """
    Create train/val/test dataloaders from pre-split file lists.

    Args:
        cfg: Hydra config

    Returns:
        Tuple of (train_loader, val_loader, test_loader, train_dataset)
    """
    selected_file = cfg.data.selected_files_json
    # print("selected_files_json =", cfg.data.selected_files_json)
    # breakpoint()

    # Check if using pre-split files
    # Original: if selected_file and any(split in selected_file for split in ['train_files.json', 'val_files.json', 'test_files.json']):
    # Modify 2026-03-17: match any train_files* (e.g. train_files_no_aug.json) not just train_files.json
    if selected_file and any(split in selected_file for split in ['train_files', 'val_files', 'test_files']):
        # Using pre-split files - load each split separately
        split_dir = Path(selected_file).parent
        # Original: train_file = str(split_dir / 'train_files.json')
        # Modify 2026-03-17: use the configured file directly (supports train_files_no_aug.json etc.)
        train_file = selected_file
        val_file = str(split_dir / 'val_files.json')
        test_file = str(split_dir / 'test_files.json')

        print(f"Loading pre-split files:")
        print(f"  Train: {train_file}")
        print(f"  Val:   {val_file}")
        print(f"  Test:  {test_file}")
        # breakpoint()

        # Create three separate datasets (NO random splitting!)
        train_dataset = create_dataset(
            data_dir=cfg.data.data_dir,
            token_pattern=cfg.data.token_pattern,
            selected_files_json=train_file,
            max_sequence_length=cfg.data.max_sequence_length,
            max_pitch=cfg.data.max_pitch,
            max_time_shift=cfg.data.max_time_shift,
            num_strings=cfg.data.num_strings,
            num_frets=cfg.data.num_frets,
            output_format=cfg.data.get("output_format", "v1"),
            max_files=cfg.data.max_files
        )

        val_dataset = create_dataset(
            data_dir=cfg.data.data_dir,
            token_pattern=cfg.data.token_pattern,
            selected_files_json=val_file,
            max_sequence_length=cfg.data.max_sequence_length,
            max_pitch=cfg.data.max_pitch,
            max_time_shift=cfg.data.max_time_shift,
            num_strings=cfg.data.num_strings,
            num_frets=cfg.data.num_frets,
            output_format=cfg.data.get("output_format", "v1"),
            max_files=None
        )

        test_dataset = create_dataset(
            data_dir=cfg.data.data_dir,
            token_pattern=cfg.data.token_pattern,
            selected_files_json=test_file,
            max_sequence_length=cfg.data.max_sequence_length,
            max_pitch=cfg.data.max_pitch,
            max_time_shift=cfg.data.max_time_shift,
            num_strings=cfg.data.num_strings,
            num_frets=cfg.data.num_frets,
            output_format=cfg.data.get("output_format", "v1"),
            max_files=None
        )

        print(f"Dataset sizes: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    else:
        # Fallback: ratio-based split (used when selected_files_json is a filter list, not pre-split)
        dataset = create_dataset(
            data_dir=cfg.data.data_dir,
            token_pattern=cfg.data.token_pattern,
            selected_files_json=selected_file,
            max_sequence_length=cfg.data.max_sequence_length,
            max_pitch=cfg.data.max_pitch,
            max_time_shift=cfg.data.max_time_shift,
            num_strings=cfg.data.num_strings,
            num_frets=cfg.data.num_frets,
            output_format=cfg.data.get("output_format", "v1"),
            max_files=cfg.data.max_files
        )

        total_size = len(dataset)
        # Modify 2026-03-17: removed breakpoint() that was blocking combined training
        # Original: breakpoint()
        train_size = int(total_size * cfg.data.train_ratio)
        val_size = int(total_size * cfg.data.val_ratio)
        test_size = total_size - train_size - val_size

        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(cfg.seed),
        )

        print(f"Dataset split (segment-level): train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    # Modify 2026-03-17: moved secondary_sources block outside the if/else so it runs
    # for BOTH ratio-split (combined) and pre-split (leduc/dadagp) modes.
    # Original: secondary_sources block was only inside the if-branch (pre-split mode).
    # Add 2026-03-17: concatenate secondary sources into train + val datasets
    secondary_sources = cfg.data.get("secondary_sources", None) or []
    if secondary_sources:
        extra_trains, extra_vals = [], []
        for src_cfg in secondary_sources:
            extra_train = create_dataset(
                data_dir=src_cfg.data_dir,
                token_pattern=src_cfg.token_pattern,
                selected_files_json=src_cfg.train_files_json,
                max_sequence_length=cfg.data.max_sequence_length,
                max_pitch=cfg.data.max_pitch,
                max_time_shift=cfg.data.max_time_shift,
                num_strings=cfg.data.num_strings,
                num_frets=cfg.data.num_frets,
                max_files=None,
            )
            extra_val = create_dataset(
                data_dir=src_cfg.data_dir,
                token_pattern=src_cfg.token_pattern,
                selected_files_json=src_cfg.val_files_json,
                max_sequence_length=cfg.data.max_sequence_length,
                max_pitch=cfg.data.max_pitch,
                max_time_shift=cfg.data.max_time_shift,
                num_strings=cfg.data.num_strings,
                num_frets=cfg.data.num_frets,
                max_files=None,
            )
            extra_trains.append(extra_train)
            extra_vals.append(extra_val)
            print(f"  + {src_cfg.data_dir}: train={len(extra_train)}, val={len(extra_val)}")

        train_dataset = ConcatDataset([train_dataset] + extra_trains)
        val_dataset   = ConcatDataset([val_dataset]   + extra_vals)
        print(f"Combined dataset sizes: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    # Modify 2026-03-17: resolve the primary TabDataset for collate_fn / vocab access
    # (needed when train_dataset / val_dataset is now a ConcatDataset)
    # Original logic handled only TabDataset and Subset (random_split); now also ConcatDataset.
    def _primary(ds):
        """Return the first underlying TabDataset regardless of wrapper type."""
        if isinstance(ds, TabDataset):
            return ds
        if isinstance(ds, ConcatDataset):
            return _primary(ds.datasets[0])
        # Subset from random_split
        return ds.dataset

    primary_train = _primary(train_dataset)
    primary_val   = _primary(val_dataset)
    primary_test  = _primary(test_dataset)

    # Create dataloaders
    # Modify 2026-03-17: pass primary_dataset so ConcatDataset loaders use correct collate_fn
    # Original:
    # train_loader = create_dataloader(train_dataset if isinstance(...) else train_dataset.dataset, ...)
    # val_loader   = create_dataloader(val_dataset   if isinstance(...) else val_dataset.dataset,   ...)
    # test_loader  = create_dataloader(test_dataset  if isinstance(...) else test_dataset.dataset,  ...)
    train_loader = create_dataloader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        primary_dataset=primary_train,
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=cfg.training.eval_batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        primary_dataset=primary_val,
    )

    test_loader = create_dataloader(
        test_dataset,
        batch_size=cfg.training.eval_batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        primary_dataset=primary_test,
    )

    # Return train_dataset for vocabulary access
    # Modify 2026-03-17: use primary_train (always a TabDataset) for vocab
    # Original: return_dataset = train_dataset if isinstance(train_dataset, TabDataset) else train_dataset.dataset
    return_dataset = primary_train
    return train_loader, val_loader, test_loader, return_dataset




# Modify 2026-03-15: also create and return LR scheduler
def create_optimizer(model: nn.Module, cfg: DictConfig):
    """Create optimizer and scheduler based on config."""
    if cfg.training.optimizer.name == "adafactor":
        from transformers import Adafactor

        optimizer = Adafactor(
            model.parameters(),
            lr=cfg.training.optimizer.lr,
            weight_decay=cfg.training.optimizer.weight_decay,
            scale_parameter=cfg.training.optimizer.scale_parameter,
            relative_step=cfg.training.optimizer.relative_step,
            warmup_init=cfg.training.optimizer.warmup_init,
        )
    elif cfg.training.optimizer.name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.training.optimizer.lr,
            weight_decay=cfg.training.optimizer.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {cfg.training.optimizer.name}")

    # Original:
    return optimizer

    # Add: create LR scheduler
    # from transformers import get_linear_schedule_with_warmup
    # scheduler = get_linear_schedule_with_warmup(
    #     optimizer,
    #     num_warmup_steps=cfg.training.scheduler.num_warmup_steps,
    #     num_training_steps=cfg.training.scheduler.num_training_steps,
    # )
    # print(f"Created LR scheduler: linear warmup over {cfg.training.scheduler.num_warmup_steps} steps, "
    #       f"total {cfg.training.scheduler.num_training_steps} steps")

    # return optimizer, scheduler


# Modify 2026-03-15: added scheduler parameter
def train_epoch(
    model: FrettingTransformer,
    train_loader: DataLoader,
    optimizer,
    # scheduler,  # Add: LR scheduler
    device: str,
    epoch: int,
    cfg: DictConfig,
):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar: tqdm[TabDatasetBatchInput] = tqdm(train_loader, desc=f"Epoch {epoch}")

    for step, batch in enumerate(pbar):
        # Move to device
        # Shapes from DataLoader (collate_fn output):
        #   input_ids: [B, L_enc] - Encoder input token IDs
        #   attention_mask: [B, L_enc] - Encoder padding mask (1=real, 0=pad)
        #   output_ids: [B, L_dec] - Full decoder sequence (for teacher forcing)
        #   decoder_attention_mask: [B, L_dec] - Decoder padding mask (1=real, 0=pad)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["output_ids"].to(device)
        decoder_attention_mask = batch["decoder_attention_mask"].to(device)

        # Shift labels right for decoder input (teacher forcing)
        # T5 expects: decoder_input = [BOS, tok1, tok2, ..., tokN-1]
        #             labels = [tok1, tok2, ..., tokN, EOS]
        # Shapes after shifting:
        #   decoder_input_ids: [B, L_dec-1] - Decoder input (all except last token)
        #   labels: [B, L_dec-1] - Target labels (all except first token)
        #   decoder_attention_mask: [B, L_dec-1] - Mask for decoder input
        decoder_input_ids = labels[:, :-1].contiguous()
        labels = labels[:, 1:].contiguous()
        
        # Mask pad tokens in labels
        labels = labels.clone()
        labels[labels == 0] = -100 # PAD id is 0
        
        decoder_attention_mask = decoder_attention_mask[:, 1:].contiguous()

        # Forward pass
        outputs: Seq2SeqLMOutput = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

        loss = outputs.loss

        # Backward pass
        loss.backward()

        # Gradient clipping
        if cfg.training.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), cfg.training.max_grad_norm
            )

        # Optimizer step
        optimizer.step()
        # scheduler.step()  # Add 2026-03-15: step LR scheduler after each optimizer update
        optimizer.zero_grad()

        # Update metrics
        total_loss += loss.item()
        num_batches += 1

        # Update progress bar
        pbar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / num_batches
    return avg_loss


def evaluate(model: FrettingTransformer, val_loader: DataLoader, device: str):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        batch: TabDatasetBatchInput
        for batch in tqdm(val_loader, desc="Evaluating"):
            # Move to device
            # Shapes: [B, L_enc], [B, L_enc], [B, L_dec], [B, L_dec]
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            output_ids = batch["output_ids"].to(device)
            decoder_attention_mask = batch["decoder_attention_mask"].to(device)

            # Shift labels right
            # Shapes after: [B, L_dec-1], [B, L_dec-1], [B, L_dec-1]
            decoder_input_ids = output_ids[:, :-1].contiguous()
            labels = output_ids[:, 1:].contiguous()
            decoder_attention_mask = decoder_attention_mask[:, 1:].contiguous()
            
             # Mask pad tokens in labels
            labels = labels.clone()
            labels[labels == 0] = -100 # PAD id is 0

            # Forward pass
            outputs: Seq2SeqLMOutput = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels,
            )

            total_loss += outputs.loss.item()
            num_batches += 1

    avg_loss = total_loss / num_batches
    return avg_loss


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig):
    """Main training function."""
    print("=" * 80)
    print("Fretting-Transformer Training")
    print("=" * 80)
    # print("\nConfiguration:")
    # print(OmegaConf.to_yaml(cfg))

    # Set seed
    set_seed(cfg.seed)

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Create output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config to output directory
    config_path = output_dir / "config.yaml"
    OmegaConf.save(cfg, config_path)
    print(f"Saved config to {config_path}")

    # Initialize training logger
    logger = TrainingLogger(log_file=output_dir / "training_log.json")
    print(f"Logging to {output_dir / 'training_log.json'}")

    # Create dataloaders
    train_loader, val_loader, test_loader, dataset = create_dataloaders(cfg)

    # Get vocabulary sizes
    input_vocab_size, output_vocab_size = dataset.get_vocab_sizes()

    # Create model
    print("\nInitializing model...")
    model = FrettingTransformer(
        input_vocab_size=input_vocab_size,
        output_vocab_size=output_vocab_size,
        model_config=OmegaConf.to_container(cfg.model),
    )
    model = model.to(device)

    # Create optimizer
    # Original: optimizer = create_optimizer(model, cfg)
    optimizer = create_optimizer(model, cfg)
    # Modify 2026-03-15: unpack (optimizer, scheduler) tuple
    # optimizer, scheduler = create_optimizer(model, cfg)

    # =========================================================================
    # CHECKPOINT RESUMPTION LOGIC
    # =========================================================================
    start_epoch = 1
    resume_path = cfg.get("checkpoint_path", None)
    if resume_path is None and "training" in cfg:
        resume_path = cfg.training.get("resume_checkpoint", None)

    if resume_path:
        resume_file = Path(resume_path)
        if resume_file.exists():
            print(f"\nLoading checkpoint from: {resume_file}")
            checkpoint = torch.load(resume_file, map_location=device)
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
                if "optimizer_state_dict" in checkpoint and optimizer is not None:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                if "epoch" in checkpoint:
                    start_epoch = checkpoint["epoch"] + 1
                print(f"Successfully resumed from epoch {start_epoch - 1}")
        else:
            print(f"\nWarning: Checkpoint not found at {resume_file}. Starting from scratch.")
    # =========================================================================

    # Training loop
    print("\nStarting training...")
    best_val_loss = float("inf")
    best_tab_accuracy = 0.0

    for epoch in range(start_epoch, cfg.training.num_epochs + 1):
        print(f"\n{'=' * 80}")
        print(f"Epoch {epoch}/{cfg.training.num_epochs}")
        print(f"{'=' * 80}")

        # Train
        # Original: train_loss = train_epoch(model, train_loader, optimizer, device, epoch, cfg)
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, cfg)
        #  Modify 2026-03-15: pass scheduler to train_epoch
        # train_loss = train_epoch(model, train_loader, optimizer, scheduler, device, epoch, cfg)
        print(f"Train loss: {train_loss:.4f}")

        # Evaluate (teacher forcing loss)
        val_loss = evaluate(model, val_loader, device)
        print(f"Val loss: {val_loss:.4f}")

        # Log epoch metrics
        logger.log_epoch(epoch=epoch, train_loss=train_loss, val_loss=val_loss)
        
        # Initialize current_tab_accuracy for this epoch
        current_tab_accuracy = None

        # Autoregressive evaluation (real accuracy)
        if cfg.training.get('ar_eval_enabled', False) and cfg.training.get('ar_eval_frequency', 0) > 0:
            if epoch % cfg.training.ar_eval_frequency == 0:
                print(f"\nRunning AR evaluation (generation + accuracy)...")
                ar_metrics: TabAccuracyMetrics | None = None
                _input_ids: torch.Tensor | None = None
                targets: torch.Tensor | None = None
                predictions: torch.Tensor | None = None
                ar_metrics, _input_ids, targets, predictions = generate_and_compute_accuracy(
                    model=model,
                    dataloader=val_loader,
                    output_vocab=dataset.output_vocab,
                    device=device,
                    max_length=cfg.training.get('ar_eval_max_length', 1024),
                    num_beams=cfg.training.get('ar_eval_num_beams', 1),
                    # max_batches=cfg.training.get('ar_eval_max_batches', None),
                    input_vocab=dataset.input_vocab, # use first N batches for quick evaluation during training
                    max_batches=None  # Use all batches for AR eval during training
                    
                )
                print(f"AR Metrics: {ar_metrics}")
                
                current_tab_accuracy = ar_metrics.tab_accuracy

                # Save generated samples safely
                preds_np = predictions.cpu().numpy() if hasattr(predictions, 'cpu') else predictions
                targets_np = targets.cpu().numpy() if hasattr(targets, 'cpu') else targets
                
                samples_file = output_dir / f"generated_samples_epoch_{epoch}.json"
                save_generated_samples(
                    predictions=preds_np,
                    targets=targets_np,
                    output_vocab=dataset.output_vocab,
                    output_file=samples_file,
                    max_samples=10
                )

                # Log AR eval to history
                logger.log_ar_eval(
                    epoch=epoch,
                    token_accuracy=ar_metrics.token_accuracy,
                    pitch_accuracy=ar_metrics.pitch_accuracy,
                    tab_accuracy=ar_metrics.tab_accuracy,
                    difficulty=ar_metrics.difficulty,
                    total_tokens=ar_metrics.total_tokens,
                    total_notes=ar_metrics.total_notes
                )

        # Save best checkpoint based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path_out = output_dir / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "config": OmegaConf.to_container(cfg),
                },
                checkpoint_path_out,
            )
            print(f"Saved best model to {checkpoint_path_out}")
        
        # Save best checkpoint based on AR tab accuracy
        # if current_tab_accuracy is not None and current_tab_accuracy > best_tab_accuracy:
        #     best_tab_accuracy = current_tab_accuracy
        #     checkpoint_path = output_dir / "best_model.pt"
        #     torch.save(
        #         {
        #             "epoch": epoch,
        #             "model_state_dict": model.state_dict(),
        #             "optimizer_state_dict": optimizer.state_dict(),
        #             "train_loss": train_loss,
        #             "val_loss": val_loss,
        #             "best_tab_accuracy": best_tab_accuracy,
        #             "config": OmegaConf.to_container(cfg),
        #         },
        #         checkpoint_path,
        #     )
        #     print(f"Saved best model to {checkpoint_path}")
        #     print(f"New best tab accuracy: {best_tab_accuracy:.4%}")

        # Save checkpoint every N epochs
        checkpoint_every_n = cfg.training.get('checkpoint_every_n_epochs', 0)
        if checkpoint_every_n > 0 and epoch % checkpoint_every_n == 0:
            checkpoint_path_out = output_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "config": OmegaConf.to_container(cfg),
                },
                checkpoint_path_out,
            )
            print(f"Saved checkpoint to {checkpoint_path_out}")

    # Final evaluation on test set
    print("\n" + "=" * 80)
    print("Final evaluation on test set")
    print("=" * 80)
    
    # Load best model checkpoint for final evaluation
    best_checkpoint_path = output_dir / "best_model.pt"
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Loaded best model from epoch {checkpoint['epoch']} for final test evaluation")
    
    test_loss = evaluate(model, test_loader, device)
    print(f"Test loss: {test_loss:.4f}")

    # Log test loss
    logger.log_test(test_loss=test_loss)

    # Final AR evaluation on test set
    if cfg.training.get('ar_eval_enabled', False):
        print(f"\nFinal AR evaluation on test set...")
        test_ar_metrics, input_ids, targets, predictions = generate_and_compute_accuracy(
            model=model,
            dataloader=test_loader,
            output_vocab=dataset.output_vocab,
            device=device,
            max_length=cfg.training.get('ar_eval_max_length', 1024),
            num_beams=cfg.training.get('ar_eval_num_beams', 1),
            max_batches=None,  # Use all batches for final evaluation
            input_vocab=dataset.input_vocab,
        )
        print(f"\nTest Set AR Metrics:")
        print(f"  Token Accuracy:  {test_ar_metrics.token_accuracy:.2%}")
        print(f"  Pitch Accuracy:  {test_ar_metrics.pitch_accuracy:.2%}")
        if test_ar_metrics.note_token_pitch_accuracy is not None:
            print(
                f"  Pitch Accuracy (NOTE_ON token):  "
                f"{test_ar_metrics.note_token_pitch_accuracy:.2%}"
            )
        print(f"  Tab Accuracy:    {test_ar_metrics.tab_accuracy:.2%}")
        print(f"  Difficulty:        {test_ar_metrics.difficulty:.2f}")

        # Save final test generated samples safely
        preds_np = predictions.cpu().numpy() if hasattr(predictions, 'cpu') else predictions
        targets_np = targets.cpu().numpy() if hasattr(targets, 'cpu') else targets
        
        samples_file = output_dir / "test_generated_samples.json"
        save_generated_samples(
            predictions=preds_np,
            targets=targets_np,
            output_vocab=dataset.output_vocab,
            output_file=samples_file,
            max_samples=20
        )

        # Log test AR eval to history
        logger.log_test_ar_eval(
            token_accuracy=test_ar_metrics.token_accuracy,
            pitch_accuracy=test_ar_metrics.pitch_accuracy,
            tab_accuracy=test_ar_metrics.tab_accuracy,
            total_tokens=test_ar_metrics.total_tokens,
            total_notes=test_ar_metrics.total_notes,
            difficulty=test_ar_metrics.difficulty
        )

    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Results saved to {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()