#!/bin/bash
cd ..
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
cd deeplab

BASEDIR=datasets/pascal_voc_seg/
MODEL=datasets/pascal_voc_seg/init_models/deeplabv3_mnv2_pascal_train_aug/model.ckpt-30000
OUTPUT=datasets/pascal_voc_seg/exp/train_on_trainaug_twoclass/
DATASET=datasets/pascal_voc_seg/tfrecord/

# Train
echo $@ | grep -q train && (
python train.py \
    --logtostderr \
    --training_number_of_steps=50000 \
    --train_split="trainaug" \
    --model_variant="mobilenet_v2" \
    --output_stride=16 \
    --train_crop_size="513,513" \
    --train_batch_size=4 \
    --dataset="personseg" \
    --tf_initial_checkpoint="${MODEL}" \
    --train_logdir="${OUTPUT}/train/" \
    --dataset_dir="${DATASET}" \
    --initalize_last_layer=False \
    --last_layers_contain_logits_only=True
)
# see https://github.com/tensorflow/models/issues/3730#issuecomment-380168917

# Evaluate
echo $@ | grep -q eval && (
ppython eval.py \
  --logtostderr \
  --eval_split="val" \
  --model_variant="mobilenet_v2" \
  --eval_crop_size="513,513" \
  --checkpoint_dir="${OUTPUT}/train/" \
  --eval_logdir="${OUTPUT}/eval/" \
  --dataset="personseg" \
  --dataset_dir="${DATASET}" \
  --max_number_of_evaluations=1
)

# Visualize the results.
echo $@ | grep -q vis && (
python vis.py \
  --logtostderr \
  --vis_split="val" \
  --model_variant="mobilenet_v2" \
  --vis_crop_size="513,513" \
  --checkpoint_dir="${OUTPUT}/train/" \
  --vis_logdir="${OUTPUT}/vis/" \
  --dataset="personseg" \
  --dataset_dir="${DATASET}" \
  --max_number_of_iterations=1
)
