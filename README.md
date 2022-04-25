# WinML_MoveNet

Requirements: Nuget: Windows.AI.MachineLearning v1.8.0

## Outline:
- Start MediaFrameReader
- Crop Camera Frame (branch Crop)
- convert MediaFrame to TensorFloat
- input TensorFloat into onnx model
- evaluate asynchronously
- draw model output


For different onnx models, load model via autogenerated .cs file. [Visual Studio tool for autogenerating .cs file with onnx model bindings](https://marketplace.visualstudio.com/items?itemName=WinML.MLGenV2) must be installed when importing different onnx models.