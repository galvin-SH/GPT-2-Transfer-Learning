// Exercise 20.5: Transfer Learning for Domain Adaptation in Rust
// Using Candle with Phi-2 model
//
// This implementation demonstrates:
// 1. Loading a pre-trained LLM (Phi-2)
// 2. Fine-tuning on domain-specific data (Rust programming)
// 3. Multiple transfer learning strategies (full fine-tuning vs layer freezing)
// 4. Performance evaluation before and after adaptation

use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::{loss, Optimizer, VarBuilder, VarMap};
use candle_transformers::models::phi::{Config as PhiConfig, Model as Phi, Model};
use hf_hub::api::sync::Api;
use tokenizers::Tokenizer;
use std::collections::HashMap;

// ============================================================================
// EVALUATION & GENERATION
// ============================================================================

/// Calculate perplexity on a test set
fn calculate_perplexity(
    model: &mut Model,
    test_inputs: &Tensor,
    test_labels: &Tensor,
) -> Result<f32> {
    let logits =model.forward(test_inputs)?;
    let (b_size, t_size, v_size) = logits.dims3()?;
    let logits_flat = logits.reshape((b_size * t_size, v_size))?;
    let labels_flat = test_labels.reshape((b_size * t_size,))?;

    let loss_val = loss::cross_entropy(&logits_flat, &labels_flat)?;
    let perplexity = loss_val.to_scalar::<f32>()?.exp();

    Ok(perplexity)
}

/// Greedy sampling for text generation
fn generate_text(
    model: &mut Model,
    tokenizer: &Tokenizer,
    device: &Device,
    prompt: &str,
    max_tokens: usize,
) -> Result<String> {
    let tokens = tokenizer
        .encode(prompt, true)
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec();
    let mut all_tokens = tokens.clone();

    for _ in 0..max_tokens {
        let tokens_tensor = Tensor::new(all_tokens.as_slice(), device)?.unsqueeze(0)?;
        let logits = model.forward(&tokens_tensor)?;

        // Get last token logits
        let seq_len = logits.dim(1)?;
        let last_token_logits = logits.narrow(1, seq_len - 1, 1)?.squeeze(1)?;

        let next_token_id = last_token_logits.argmax(0)?.to_scalar::<u32>()?;
        all_tokens.push(next_token_id);
    }

    tokenizer.decode(&all_tokens, false).map_err(anyhow::Error::msg)
}

/// Evaluate model on multiple prompts
fn evaluate_on_prompts(
    model: &mut Phi,
    tokenizer: &Tokenizer,
    device: &Device,
    prompts: &[&str],
) {
    println!("\n=== Model Evaluation ===");
    for prompt in prompts {
        println!("\nPrompt: \"{}\"", prompt);
        match generate_text(model, tokenizer, device, prompt, 30) {
            Ok(text) => println!("Generated: \"{}\"", text),
            Err(e) => println!("Error: {}", e),
        }
    }
}

// ============================================================================
// DATASET PREPARATION
// ============================================================================

fn prepare_domain_dataset() -> String {
    // Domain-specific dataset: Rust programming concepts
    r#"
    In Rust, ownership is a set of rules that govern memory management.
    The Vec<T> type is a growable array stored on the heap.
    Result<T, E> is used for error handling in Rust programs.
    The match expression enables pattern matching on enums.
    Traits define shared behavior in an abstract way.
    The borrow checker ensures memory safety at compile time.
    Cargo is the Rust package manager and build system.
    impl blocks define methods on structs and enums.
    The Option<T> enum handles the presence or absence of values.
    Lifetimes ensure references are valid for their entire scope.
    The String type is a growable UTF-8 encoded string.
    Closures are anonymous functions that can capture their environment.
    The derive attribute automatically implements traits.
    Async/await syntax enables asynchronous programming.
    The candle crate provides tensor operations for machine learning.
    Error propagation uses the ? operator for Result types.
    Generic types enable code reuse with type parameters.
    The println! macro prints formatted text to stdout.
    References borrow values without taking ownership.
    Mutable references allow modification of borrowed data.
    "#.to_string()
}

fn prepare_test_dataset() -> String {
    // Separate test set for evaluation
    r#"
    Rust's ownership system prevents data races at compile time.
    The HashMap<K, V> type stores key-value pairs.
    Iterators provide a way to process sequences of elements.
    The std::fs module provides file system operations.
    Smart pointers like Box<T> enable heap allocation.
    "#.to_string()
}

fn tokenize_dataset(
    text: &str,
    tokenizer: &Tokenizer,
    device: &Device,
    block_size: usize,
) -> Result<(Tensor, Tensor)> {
    let tokens = tokenizer
        .encode(text, true)
        .map_err(anyhow::Error::msg)?
        .get_ids()
        .to_vec();

    let actual_block_size = block_size.min(tokens.len() - 1);
    let inputs_vec: Vec<u32> = tokens[..actual_block_size].to_vec();
    let labels_vec: Vec<u32> = tokens[1..=actual_block_size].to_vec();

    let inputs = Tensor::new(inputs_vec.as_slice(), device)?.unsqueeze(0)?;
    let labels = Tensor::new(labels_vec.as_slice(), device)?.unsqueeze(0)?;

    Ok((inputs, labels))
}

// ============================================================================
// TRANSFER LEARNING STRATEGIES
// ============================================================================

enum TransferStrategy {
    FullFineTune,           // Train all parameters
    FreezeLowerLayers,      // Freeze first 50% of layers
    FreezeEmbeddings,       // Only freeze embedding layers
    AdapterLayers,          // Freeze all, train only specific layers (simulated)
}

fn apply_transfer_strategy(
    varmap: &VarMap,
    strategy: &TransferStrategy,
    n_layers: usize,
) -> Vec<candle_core::Var> {
    let all_vars = varmap.all_vars();

    match strategy {
        TransferStrategy::FullFineTune => {
            println!("Strategy: Full Fine-Tuning (all {} parameters)", all_vars.len());
            all_vars
        }
        TransferStrategy::FreezeLowerLayers => {
            println!("Strategy: Freeze Lower Layers (freeze first {})", n_layers / 2);
            all_vars.into_iter()
                .filter(|var| {
                    let name = var.as_tensor().to_string();
                    // Only train upper layers (h.6 onwards for 12-layer model)
                    !name.contains(&format!("h.{}", 0)) &&
                        !name.contains(&format!("h.{}", 1)) &&
                        !name.contains(&format!("h.{}", 2)) &&
                        !name.contains(&format!("h.{}", 3)) &&
                        !name.contains(&format!("h.{}", 4)) &&
                        !name.contains(&format!("h.{}", 5))
                })
                .collect()
        }
        TransferStrategy::FreezeEmbeddings => {
            println!("Strategy: Freeze Embeddings Only");
            all_vars.into_iter()
                .filter(|var| {
                    let name = var.as_tensor().to_string();
                    // Freeze embedding layers
                    !name.contains("wte") && !name.contains("wpe")
                })
                .collect()
        }
        TransferStrategy::AdapterLayers => {
            println!("Strategy: Adapter Layers (train last 2 layers + head)");
            all_vars.into_iter()
                .filter(|var| {
                    let name = var.as_tensor().to_string();
                    // Only train last layers and LM head
                    name.contains("ln_f") ||
                        name.contains("lm_head") ||
                        name.contains(&format!("h.{}", n_layers - 1)) ||
                        name.contains(&format!("h.{}", n_layers - 2))
                })
                .collect()
        }
    }
}

// ============================================================================
// TRAINING LOOP
// ============================================================================

fn train_model(
    model: &mut Model,
    train_data: &(Tensor, Tensor),
    test_data: &(Tensor, Tensor),
    trainable_params: Vec<candle_core::Var>,
    learning_rate: f64,
    epochs: usize,
) -> Result<Vec<f32>> {
    let mut optimizer = candle_nn::AdamW::new_lr(trainable_params.clone(), learning_rate)?;
    let mut losses = Vec::new();

    println!("\nTraining with {} trainable parameters", trainable_params.len());
    println!("Epochs: {}, Learning Rate: {}", epochs, learning_rate);
    println!("\n{:<8} {:<12} {:<12}", "Epoch", "Train Loss", "Perplexity");
    println!("{}", "-".repeat(35));

    for epoch in 0..epochs {
        // Forward pass
        let logits = model.forward(&train_data.0)?;
        let (b_size, t_size, v_size) = logits.dims3()?;
        let logits_flat = logits.reshape((b_size * t_size, v_size))?;
        let labels_flat = train_data.1.reshape((b_size * t_size,))?;

        // Calculate loss
        let loss_val = loss::cross_entropy(&logits_flat, &labels_flat)?;
        let loss_scalar = loss_val.to_scalar::<f32>()?;
        losses.push(loss_scalar);

        // Backward pass
        optimizer.backward_step(&loss_val)?;

        // Evaluate every 5 epochs
        if epoch % 5 == 0 || epoch == epochs - 1 {
            let perplexity = calculate_perplexity(model, &test_data.0, &test_data.1)?;
            println!("{:<8} {:<12.6} {:<12.2}", epoch, loss_scalar, perplexity);
        }
    }

    Ok(losses)
}

// ============================================================================
// MAIN EXPERIMENT
// ============================================================================

fn main() -> Result<()> {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║  Exercise 20.5: Transfer Learning for Domain Adaptation   ║");
    println!("║  Using Candle + Phi-2 for Rust Programming Domain         ║");
    println!("╚════════════════════════════════════════════════════════════╝\n");

    // ========================================================================
    // Setup
    // ========================================================================
    println!("🔧 Setting up environment...");
    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("   Device: {:?}", device);

    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    // ========================================================================
    // Load Pre-trained Model
    // ========================================================================
    println!("\n📦 Loading pre-trained Phi-2 model...");
    let api = Api::new()?;
    let repo = api.model("microsoft/phi-2".to_string());

    let tokenizer_file = repo.get("tokenizer.json")?;
    let tokenizer = Tokenizer::from_file(tokenizer_file).map_err(anyhow::Error::msg)?;
    println!("   ✓ Tokenizer loaded");

    let config_file = repo.get("config.json")?;
    let config: PhiConfig = serde_json::from_str(&std::fs::read_to_string(config_file)?)?;
    println!("   ✓ Config loaded");

    let weights_file = repo.get("model.safetensors")?;
    varmap.load(&weights_file)?;
    println!("   ✓ Weights loaded");

    let mut model = Phi::new(&config, vb)?;
    println!("   ✓ Model initialized");

    // ========================================================================
    // Prepare Datasets
    // ========================================================================
    println!("\n📚 Preparing domain-specific datasets...");
    let train_text = prepare_domain_dataset();
    let test_text = prepare_test_dataset();

    let block_size = 128;
    let train_data = tokenize_dataset(&train_text, &tokenizer, &device, block_size)?;
    let test_data = tokenize_dataset(&test_text, &tokenizer, &device, 64)?;
    println!("   Train shape: {:?}", train_data.0.shape());
    println!("   Test shape: {:?}", test_data.0.shape());

    // ========================================================================
    // Baseline Evaluation (Before Fine-tuning)
    // ========================================================================
    println!("\n📊 Baseline Evaluation (Pre-trained model)");
    let baseline_perplexity = calculate_perplexity(&mut model, &test_data.0, &test_data.1)?;
    println!("   Initial Perplexity: {:.2}", baseline_perplexity);

    let eval_prompts = vec![
        "In Rust, Vec<T> is",
        "The Result type in Rust",
        "Ownership in Rust means",
    ];
    evaluate_on_prompts(&mut model, &tokenizer, &device, &eval_prompts);

    // Save weights file path for reloading
    let weights_path = weights_file.clone();

    // ========================================================================
    // Experiment with Different Transfer Learning Strategies
    // ========================================================================
    println!("\n\n╔════════════════════════════════════════════════════════════╗");
    println!("║           TRANSFER LEARNING STRATEGY COMPARISON           ║");
    println!("╚════════════════════════════════════════════════════════════╝");

    let strategies = vec![
        ("Full Fine-Tuning", TransferStrategy::FullFineTune),
        ("Freeze Lower Layers", TransferStrategy::FreezeLowerLayers),
        ("Freeze Embeddings", TransferStrategy::FreezeEmbeddings),
        ("Adapter Layers", TransferStrategy::AdapterLayers),
    ];

    let mut results: HashMap<String, (Vec<f32>, f32)> = HashMap::new();

    for (name, strategy) in strategies {
        println!("\n{}", "=".repeat(60));
        println!(" {} ", name);
        println!("{}", "=".repeat(60));

        // Create fresh VarMap and model for each strategy
        let mut strategy_varmap = VarMap::new();
        strategy_varmap.load(&weights_path)?;
        let strategy_vb = VarBuilder::from_varmap(&strategy_varmap, DType::F32, &device);
        let mut fresh_model = Phi::new(&config, strategy_vb)?;

        // Apply strategy
        let trainable_params = apply_transfer_strategy(&strategy_varmap, &strategy, 12);

        // Train
        let losses = train_model(
            &mut fresh_model,
            &train_data,
            &test_data,
            trainable_params,
            1e-4,
            20,
        )?;

        // Final evaluation
        let final_perplexity = calculate_perplexity(&mut fresh_model, &test_data.0, &test_data.1)?;
        println!("\n✓ Final Perplexity: {:.2}", final_perplexity);

        results.insert(name.to_string(), (losses, final_perplexity));

        // Generate samples
        evaluate_on_prompts(&mut fresh_model, &tokenizer, &device, &eval_prompts);
    }

    // ========================================================================
    // Final Results Summary
    // ========================================================================
    println!("\n\n╔════════════════════════════════════════════════════════════╗");
    println!("║                    RESULTS SUMMARY                         ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!("\nBaseline (Pre-trained): {:.2} perplexity\n", baseline_perplexity);
    println!("{:<25} {:<15} {:<15}", "Strategy", "Final Perplexity", "Improvement");
    println!("{}", "-".repeat(60));

    for (strategy, (_, perplexity)) in results.iter() {
        let improvement = ((baseline_perplexity - perplexity) / baseline_perplexity) * 100.0;
        println!("{:<25} {:<15.2} {:<15.1}%", strategy, perplexity, improvement);
    }

    println!("\n\n🎓 Key Findings:");
    println!("   • Full fine-tuning typically achieves lowest perplexity");
    println!("   • Freezing strategies reduce training time & overfitting risk");
    println!("   • Adapter layers offer good trade-off for limited data");
    println!("   • Domain adaptation successful with minimal training data\n");

    Ok(())
}