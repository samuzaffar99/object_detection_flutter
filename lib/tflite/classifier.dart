import 'dart:convert';
import 'dart:math';
import 'dart:ui';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as image_lib;
import 'package:object_detection/tflite/recognition.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';

import 'stats.dart';

/// Classifier
class Classifier {
  /// Instance of Interpreter
  Interpreter? _interpreter;

  /// Labels file loaded as list
  List<String>? _labels;

  static const String modelFileName = "efficientnet/EfficientNetUint8.tflite";
  static const String labelFileName = "efficientnet/Classes.txt";

  /// Input size of image (height = width = 300)
  static const int inputSize = 224;

  /// Result score threshold
  static const double threshold = 0.8;

  /// [ImageProcessor] used to pre-process the image
  ImageProcessor? imageProcessor;

  /// Padding the image to transform into square
  int? padSize;

  /// Shapes of output tensors
  List<List<int>> _outputShapes = [];

  /// Types of output tensors
  List<TfLiteType> _outputTypes = [];

  Classifier({
    Interpreter? interpreter,
    List<String>? labels,
  }) {
    loadModel(interpreter: interpreter);
    loadLabels(labels: labels);
  }

  /// Loads interpreter from asset
  void loadModel({Interpreter? interpreter}) async {
    try {
      // print("try load interpreter");
      final gpuDelegateV2 = GpuDelegateV2(
        options: GpuDelegateOptionsV2(
            isPrecisionLossAllowed: true,
            // TfLiteGpuInferenceUsage.fastSingleAnswer,
            // TfLiteGpuInferencePriority.minLatency,
            // TfLiteGpuInferencePriority.auto,
            // TfLiteGpuInferencePriority.auto,
            // experimentalFlags: const [TfLiteGpuExperimentalFlags.none],
            ),
      );
      InterpreterOptions interpreterOptions = InterpreterOptions()
        ..threads = 4
        // ..useNnApiForAndroid = true
        ..addDelegate(gpuDelegateV2)
      ;
      _interpreter = interpreter ??
          await Interpreter.fromAsset(
            modelFileName,
            options: interpreterOptions,
            // ..useFlexDelegateAndroid = true,
          );
      // print("loaded interpreter");
      var outputTensors = _interpreter!.getOutputTensors();
      _outputShapes = [];
      _outputTypes = [];
      for (var tensor in outputTensors) {
        _outputShapes.add(tensor.shape);
        _outputTypes.add(tensor.type);
      }
    } catch (e) {
      print("Error while creating interpreter: $e");
    }
  }

  /// Loads labels from assets
  void loadLabels({List<String>? labels}) async {
    try {
      _labels = labels ?? await FileUtil.loadLabels("assets/" + labelFileName);
    } catch (e) {
      print("Error while loading labels: $e");
    }
  }

  /// Pre-process the image
  TensorImage getProcessedImage(TensorImage inputImage) {
    padSize = max(inputImage.height, inputImage.width);
    imageProcessor ??= ImageProcessorBuilder()
        .add(ResizeWithCropOrPadOp(padSize!, padSize!))
        .add(ResizeOp(inputSize, inputSize, ResizeMethod.BILINEAR))
        // .add(NormalizeOp(127.5, 127.5))
        // .add(DequantizeOp(128.0, 1 / 128.0))
        // .add(QuantizeOp(0,1))
        .build();
    inputImage = imageProcessor!.process(inputImage);
    return inputImage;
  }

  /// Runs object detection on the input image
  Map<String, dynamic>? predict(image_lib.Image image) {
    var predictStartTime = DateTime.now().millisecondsSinceEpoch;

    if (_interpreter == null) {
      print("Interpreter not initialized");
      return null;
    }

    var preProcessStart = DateTime.now().millisecondsSinceEpoch;

    // Create TensorImage from image
    TensorImage inputImage = TensorImage.fromImage(image);

    // Pre-process TensorImage
    try {
      inputImage = getProcessedImage(inputImage);
    } on Exception catch (e) {
      print(e);
    }

    var preProcessElapsedTime =
        DateTime.now().millisecondsSinceEpoch - preProcessStart;

    // TensorBuffers for output tensors
    TensorBuffer outputScores = TensorBufferFloat(_outputShapes[0]);

    // Use [TensorImage.buffer] or [TensorBuffer.buffer] to pass by reference
    List<Object> inputs = [inputImage.buffer];

    // Outputs map
    Map<int, Object> outputs = {
      0: outputScores.buffer,
    };

    var inferenceTimeStart = DateTime.now().millisecondsSinceEpoch;
    // print(inputs);
    // print(outputs);

    // print(inputImage.tensorBuffer);

    // print("check1");
    // run inference
    try {
      _interpreter!.runForMultipleInputs(inputs, outputs);
    } on Exception catch (e) {
      print(e);
    }
    // print("pass");
    var inferenceTimeElapsed =
        DateTime.now().millisecondsSinceEpoch - inferenceTimeStart;

    List<Recognition> recognitions = [];
    final List<double> scoreList = outputScores.getDoubleList();

    final int maxIndex = scoreList.indexOf(scoreList.reduce(max));
    final double score = scoreList[maxIndex];
    if (score > threshold) {
      print(score);
      // Using labelOffset = 1 as ??? at index 0
      int labelOffset = 0;
      var labelIndex = maxIndex + labelOffset;
      print(labelIndex);

      final labels = [
        'Bicycle',
        'Bicycle Stand',
        'Electric Scooter',
        'Golf cart',
        'Gym Equipment',
        'Sign Board',
        'bench',
        'boat',
        'building',
        'chair',
        'flower',
        'palm tree',
        'stairs',
        'table',
        'tree',
      ];
      // var label = _labels!.elementAt(labelIndex);
      var label = labels.elementAt(labelIndex);
      print(label);
      if (label != "?") {
        Rect transformedRect = Rect.fromCenter(
            center: const Offset(0, 0), width: 240, height: 240);
        recognitions.add(
          Recognition(maxIndex, label, score, transformedRect),
        );
      }
    }

    var predictElapsedTime =
        DateTime.now().millisecondsSinceEpoch - predictStartTime;

    return {
      "recognitions": recognitions,
      "stats": Stats(
          totalPredictTime: predictElapsedTime,
          inferenceTime: inferenceTimeElapsed,
          preProcessingTime: preProcessElapsedTime)
    };
  }

  /// Gets the interpreter instance
  Interpreter? get interpreter => _interpreter;

  /// Gets the loaded labels
  List<String>? get labels => _labels;
}
