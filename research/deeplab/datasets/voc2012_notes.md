pascal_voc_seg + augmentation:

SegmentationClass/ #2913
	#2913 ImageSets/Segmentation/trainval.txt (subset1u2)
	#1464 ImageSets/Segmentation/train.txt (subset1)
	#1449 ImageSets/Segmentation/val.txt (subset2)
SegmentationClassAug/ #12031 (subset1u2u3)
	#10582 ImageSets/Segmentation/trainaug.txt (subset1u3)

mIOU for "person" with original data (trainval set):
  ~0.8357 basically without training
  ~0.8856 after 100 000 epochs
