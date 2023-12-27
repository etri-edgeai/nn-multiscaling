# Image Classification Task Builder

This module provides a TaskBuilder class for image classification tasks. It is designed to facilitate the loading of datasets, preprocessing of models, and model compilation.

## Requirements

- TensorFlow
- Bespoke module for task building
- NNCompress module for neural network compression

## Configuration

Instantiate the ImageClassificationBuilder with a configuration dictionary containing dataset parameters, preprocessing options, and model compilation settings.

Example:
```python
config = {
    "dataset": "your_dataset_name",
    "width": 224,
    "batch_size": 32,
    "augment": True,
    "n_classes": 10,
    "sampling_ratio": 1.0,
    "cutmix_alpha": 0.0,
    "mixup_alpha": 0.0,
    "use_amp": False
}

task_builder = ImageClassificationBuilder(config)
```

## Methods

- `load_dataset(split, is_tl)`: Loads the dataset for a given split ('train', 'val', 'test') and applies transfer learning preprocessing if `is_tl` is True.
- `prep(model, is_teacher, for_benchmark)`: Prepares the model by setting the data type for mixed precision, removing or adding data augmentation, and adjusting the batch size if necessary.
- `compile(model, mode, run_eagerly)`: Compiles the model for evaluation with the Adam optimizer and categorical crossentropy loss.

## Customization

You can customize the TaskBuilder by altering the configuration dictionary to match the specifics of your dataset and model requirements.

## Contributing

To contribute to the TaskBuilder, you can extend its functionality to support more modes, optimizers, and dataset types. Ensure that any added custom layers or functions are properly integrated into the bespoke and nncompress modules.