CUDA_VISIBLE_DEVICES=0 python platerec/main.py \
--train_data "/media/ps/0A9AD66165F33762/XPC/plate_all/train" \
--valid_data "/media/ps/0A9AD66165F33762/XPC/plate_all/valid" \
--output_dir "/media/ps/A/XPC/output/PlateRec/1" \
--batch_size 512 \
--num_classes 84 \
--arch "resnet18" \
--imgH 64 \
--imgW 128