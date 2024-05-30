python train_ensemble.py --epochs 100 --model_name EarlyEnsemble --use_efficient True  --use_mobile True > ./logs/BRACOL_EarlyEnsemble_True_True_False.log
python train_ensemble.py --epochs 100 --model_name EfficientNet > ./logs/BRACOL_EfficientNet.log
python train_ensemble.py --epochs 100 --model_name MobileNet > ./logs/BRACOL_MobileNet.log
python train_ensemble.py --epochs 100 --model_name ViT > ./logs/BRACOL_ViT.log
python train_ensemble.py --epochs 100 --model_name ResNet50 > ./logs/BRACOL_ResNet50.log
python train_ensemble.py --epochs 100 --model_name EarlyEnsemble --use_efficient True --use_mobile True --use_vit True > ./logs/BRACOL_EarlyEnsemble_True_True_True.log
python train_ensemble.py --epochs 100 --model_name EarlyEnsemble --use_efficient True --use_vit True > ./logs/BRACOL_EarlyEnsemble_True_False_True.log
python train_ensemble.py --epochs 100 --model_name EarlyEnsemble --use_mobile True --use_vit True > ./logs/BRACOL_EarlyEnsemble_False_True_True.log
