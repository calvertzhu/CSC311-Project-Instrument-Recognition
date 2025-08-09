import torch
from models.vgg_cnn import create_irmas_model, IRMASConfig
from models.trainer import IRMASTrainer
from pathlib import Path

def test_saved_model(model_path="models/saved_models/irmas_nn_best.pth"):
    """Load and test saved model on test dataset."""
    
    print("TESTING SAVED IRMAS MODEL")
    print("=" * 40)
    
    # Create model
    config = IRMASConfig()
    model = create_irmas_model(config)
    
    # Load saved weights
    model_file = Path(model_path)
    if not model_file.exists():
        print(f"ERROR: Model file {model_path} not found!")
        print("Available models:")
        saved_dir = Path("models/saved_models")
        for file in saved_dir.glob("*.pth"):
            print(f"  - {file.name}")
        return
    
    print(f"Loading model: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    
    # Create trainer (just for test evaluation)
    trainer = IRMASTrainer(
        model=model,
        model_name="loaded_irmas_nn",
        device='auto'
    )
    
    # Evaluate on test dataset
    print("\nEvaluating on test dataset...")
    test_results = trainer.evaluate_test()
    
    print("\nTEST RESULTS:")
    print("=" * 30)
    for metric, value in test_results.items():
        if isinstance(value, (int, float)):
            print(f"{metric}: {value:.4f}")
        else:
            print(f"{metric}: {value}")

if __name__ == "__main__":
    test_saved_model() 