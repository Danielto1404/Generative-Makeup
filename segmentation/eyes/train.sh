python3 train.py --annotation_path=/Users/a19378208/Documents/GitHub/Generative-Makeup/data/50-labels.json \
                 --images_root=/Users/a19378208/Documents/GitHub/Generative-Makeup/data/images \
                 --model_path=checkpoints/model.pth \
                 --num_classes=3 \
                 --num_epochs=10 \
                 --batch_size=4 \
                 --lr=0.001 \
                 --l2=0.001 \
                 --device=cpu