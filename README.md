# AnnotationEfficient-CellCounting
This is the implementation for the paper, _Annotation-efficient Cell Counting_, (MICCAI 2021).

- To run the code:
> python VGG_main.py /LABELED/JSON_FILE /UNLABELED/JSON_FILE /VAL/JSON_FILE GPU_ID SAVED_NAME
- For example:
- python VGG_main.py ./VGG_sa_n10_train_labeled.json ./VGG_sa_n10_train_unlabeled.json ./VGG_sa_n10_val.json 0 VGG_sa_n10_model_
