
from ultralytics import YOLO

# Load a model
model = YOLO("yolo11-mobilevit-bifpn-bra.yaml").load("yolo11x.pt")  # load a pretrained model. change this to the model you want to use, n, s, l etc. (only ending)

# Train the model with optimized hyperparameters for waste detection
model.train(
    data="datafinal.yaml", 
    epochs=100,  # Increased epochs for better convergence
    lr0=0.001,   # Better initial learning rate
    lrf=0.01,    # Final learning rate factor
    weight_decay=0.0005,  # Standard weight decay
    optimizer='AdamW', 
    warmup_epochs=3,  # Add warmup for stability
    mosaic=1.0, 
    close_mosaic=10,  # Close mosaic earlier for final training stability
    batch=16,    # Reasonable batch size
    imgsz=640,   # Standard image size
    patience=50, # Early stopping patience
    save_period=10,  # Save checkpoints every 10 epochs
    val=True,    # Enable validation
    plots=True,  # Generate training plots
)
