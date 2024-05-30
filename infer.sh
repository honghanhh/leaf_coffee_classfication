python inference.py --model_path ./checkpoints/MobileNet/False_False_False_bestacc.pth --dataset bracol --rootdir ./data/BRACOL/symptom/test/
python inference.py --model_path ./checkpoints/EfficientNet/False_False_False_bestacc.pth --dataset bracol --rootdir ./data/BRACOL/symptom/test/
python inference.py --model_path ./checkpoints/ViT/False_False_False_bestacc.pth --dataset bracol --rootdir ./data/BRACOL/symptom/test/
python inference.py --model_path ./checkpoints/EarlyEnsemble/True_True_True_bestacc.pth --dataset bracol --rootdir ./data/BRACOL/symptom/test/
python inference.py --model_path ./checkpoints/EarlyEnsemble/True_True_False_bestacc.pth --dataset bracol --rootdir ./data/BRACOL/symptom/test/
python inference.py --model_path ./checkpoints/EarlyEnsemble/True_False_True_bestacc.pth --dataset bracol --rootdir ./data/BRACOL/symptom/test/
python inference.py --model_path ./checkpoints/EarlyEnsemble/False_True_True_bestacc.pth --dataset bracol --rootdir ./data/BRACOL/symptom/test/
