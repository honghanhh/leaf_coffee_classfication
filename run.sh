python train_ensemble.py --epochs 100 --model_name EarlyEnsemble --use_efficient True  --use_mobile True
python train_ensemble.py --epochs 100 --model_name EfficientNet
python train_ensemble.py --epochs 100 --model_name MobileNet
python train_ensemble.py --epochs 100 --model_name ViT
python train_ensemble.py --epochs 100 --model_name ResNet50
python train_ensemble.py --epochs 100 --model_name EarlyEnsemble --use_efficient True --use_mobile True --use_vit True
python train_ensemble.py --epochs 100 --model_name EarlyEnsemble --use_efficient True --use_vit True
python train_ensemble.py --epochs 100 --model_name EarlyEnsemble --use_mobile True --use_vit True