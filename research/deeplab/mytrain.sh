#!/bin/bash
cd ..
export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
cd deeplab

BASEDIR=datasets/pascal_voc_seg/
MODEL=datasets/pascal_voc_seg/init_models/deeplabv3_mnv2_pascal_train_aug/model.ckpt-30000
OUTPUT=datasets/pascal_voc_seg/exp/train_on_trainaugval_twoclass/
DATASET=datasets/pascal_voc_seg/tfrecord/

# Train
echo $@ | grep -q train && (
python train.py \
    --logtostderr \
    --training_number_of_steps=100000 \
    --train_split="trainaugval" \
    --model_variant="mobilenet_v2" \
    --output_stride=16 \
    --train_crop_size="513,513" \
    --train_batch_size=16 \
    --dataset="personseg" \
    --tf_initial_checkpoint="${MODEL}" \
    --train_logdir="${OUTPUT}/train/" \
    --dataset_dir="${DATASET}" \
    --base_learning_rate=0.01 \
    --initialize_last_layer="false" \
    --last_layers_contain_logits_only="true"
)
# see https://github.com/tensorflow/models/issues/3730#issuecomment-380168917

# Evaluate
echo $@ | grep -q eval && (
python eval.py \
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

# export network
echo $@ | grep -q export && (
python export_model.py \
  --logtostderr \
  --checkpoint_path="${OUTPUT}/train/model.ckpt-100000" \
  --export_path="${OUTPUT}/frozen_inference_graph.pb" \
  --model_variant="mobilenet_v2" \
  --num_classes=4 \
  --crop_size=513 \
  --crop_size=513 \
  --inference_scales=1.0
)
