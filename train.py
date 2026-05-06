import wandb
from wandb.integration.ultralytics import add_wandb_callback
from ultralytics import YOLO

wandb.init(project="Fashion-FastSAM-Novel", name="H100-Finetune-v2-fixed")
model = YOLO('FastSAM-s.pt')
add_wandb_callback(model, enable_model_checkpointing=True)
model.train(
    data='path/to/fashion.yaml',
    epochs=100,         
    imgsz=640,
    batch=32,          
    lr0=1e-3,          
    lrf=0.01,
    cos_lr=True,        
    device=1,          
    workers=16,        
    patience=15,       
    save=True,
    plots=True,
    overlap_mask=True,
    mask_ratio=1,      
    name='FASTSAM-S_IMATERIALISTICC_FINAL_XVAR',
    project='fashion_fastsam_BLACK_RED_test',
)
wandb.finish()