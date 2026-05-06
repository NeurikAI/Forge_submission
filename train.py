import wandb
from wandb.integration.ultralytics import add_wandb_callback
from ultralytics import YOLO

wandb.init(project="Fashion-FastSAM-Novel", name="H100-Finetune-v2-fixed")
model = YOLO('FastSAM-s.pt')
add_wandb_callback(model, enable_model_checkpointing=True)

model.train(
    data='fashion.yaml',
    epochs=20,
    imgsz=640,
    batch=16,  
    lr0 =1e-3,
    lrf= 0.01,    
    device=1,          
    workers=8,        
    patience=15,       
    amp=True,          
    save=True,
    plots=True,
    overlap_mask=True,
    mask_ratio=1,
    name='fashion_finetune',
    project='fashion_fastsam',
    cache=False,
)
wandb.finish()