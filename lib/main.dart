import 'package:flutter/material.dart';
import 'package:flutter/foundation.dart';
import 'package:camera/camera.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img;

class IsolateData {
  final CameraImage cameraImage;
  final int interpreterAddress;
  final List<String> labels;
  final int inputSize;

  IsolateData(
    this.cameraImage,
    this.interpreterAddress,
    this.labels,
    this.inputSize,
  );
}

Future<List<Map<String, dynamic>>> runModelOnIsolate(IsolateData isolateData) async {
  final interpreter = Interpreter.fromAddress(isolateData.interpreterAddress);
  final inputSize = isolateData.inputSize;
  final inputTensor = _preprocessCameraImage(isolateData.cameraImage, inputSize);
  final outputShape = interpreter.getOutputTensor(0).shape;
  final outputTensor = List.filled(outputShape.reduce((a, b) => a * b), 0.0).reshape(outputShape);
  interpreter.run(inputTensor, outputTensor);
  final recognitions = _postprocessOutput(outputTensor[0], isolateData.labels);
  return recognitions;
}

List<List<List<List<double>>>> _preprocessCameraImage(CameraImage image, int inputSize) {
  final img.Image convertedImage = img.Image(width: image.width, height: image.height);
  final int uvRowStride = image.planes[1].bytesPerRow;
  final int? uvPixelStride = image.planes[1].bytesPerPixel;

  for (int y = 0; y < image.height; y++) {
    for (int x = 0; x < image.width; x++) {
      final int uvIndex = uvPixelStride! * (x / 2).floor() + uvRowStride * (y / 2).floor();
      final int index = y * image.width + x;
      final yp = image.planes[0].bytes[index];
      final up = image.planes[1].bytes[uvIndex];
      final vp = image.planes[2].bytes[uvIndex];
      int r = (yp + vp * 1436 / 1024 - 179).round().clamp(0, 255);
      int g = (yp - up * 46549 / 131072 + 44 - vp * 93604 / 131072 + 91).round().clamp(0, 255);
      int b = (yp + up * 1814 / 1024 - 227).round().clamp(0, 255);
      convertedImage.setPixelRgba(x, y, r, g, b, 255);
    }
  }
  
  final resizedImage = img.copyResize(convertedImage, width: inputSize, height: inputSize);
  final imageMatrix = List.generate(
    inputSize,
    (y) => List.generate(
      inputSize,
      (x) {
        final pixel = resizedImage.getPixel(x, y);
        return [pixel.r / 255.0, pixel.g / 255.0, pixel.b / 255.0];
      },
    ),
  );
  return [imageMatrix];
}

List<Map<String, dynamic>> _postprocessOutput(List<dynamic> output, List<String> labels) {
  final List<List<double>> transposedOutput = List.generate(
    output[0].length, (i) => List.generate(output.length, (j) => output[j][i].toDouble())
  );

  List<Map<String, dynamic>> candidates = [];
  const double confidenceThreshold = 0.5;

  for (final detection in transposedOutput) {
    final scores = detection.sublist(4);
    double maxScore = 0;
    int bestClassIndex = -1;

    for (int i = 0; i < scores.length; i++) {
      if (scores[i] > maxScore) {
        maxScore = scores[i];
        bestClassIndex = i;
      }
    }

    if (maxScore > confidenceThreshold && bestClassIndex != -1 && bestClassIndex < labels.length) {
      final cx = detection[0];
      final cy = detection[1];
      final w = detection[2];
      final h = detection[3];
      final left = (cx - w / 2);
      final top = (cy - h / 2);
      final right = (cx + w / 2);
      final bottom = (cy + h / 2);

      candidates.add({
        "rect": Rect.fromLTRB(left, top, right, bottom),
        "label": labels[bestClassIndex],
        "score": maxScore,
      });
    }
  }
  return _nonMaxSuppression(candidates);
}

List<Map<String, dynamic>> _nonMaxSuppression(List<Map<String, dynamic>> boxes) {
  if (boxes.isEmpty) return [];
  List<Map<String, dynamic>> picked = [];
  const double iouThreshold = 0.4;
  boxes.sort((a, b) => (b["score"] as double).compareTo(a["score"] as double));

  while (boxes.isNotEmpty) {
    final current = boxes.removeAt(0);
    picked.add(current);
    boxes.removeWhere((box) {
      final double iou = _calculateIoU(current["rect"], box["rect"]);
      return iou > iouThreshold;
    });
  }
  return picked;
}

double _calculateIoU(Rect a, Rect b) {
  final double xA = a.left > b.left ? a.left : b.left;
  final double yA = a.top > b.top ? a.top : b.top;
  final double xB = a.right < b.right ? a.right : b.right;
  final double yB = a.bottom < b.bottom ? a.bottom : b.bottom;
  final double intersectionArea = (xB - xA).clamp(0, double.infinity) * (yB - yA).clamp(0, double.infinity);
  final double aArea = (a.right - a.left) * (a.bottom - a.top);
  final double bArea = (b.right - b.left) * (b.bottom - a.top);
  final double unionArea = aArea + bArea - intersectionArea;
  return unionArea > 0 ? intersectionArea / unionArea : 0;
}

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final cameraPermissionStatus = await Permission.camera.request();
  if (cameraPermissionStatus.isGranted) {
    try {
      final cameras = await availableCameras();
      if (cameras.isNotEmpty) {
        runApp(MyApp(camera: cameras.first));
      } else {
        runApp(const ErrorApp("Tidak ada kamera yang ditemukan di perangkat ini."));
      }
    } catch (e) {
        runApp(ErrorApp("Gagal mendapatkan daftar kamera: $e"));
    }
  } else {
    runApp(const ErrorApp("Izin kamera ditolak. Aplikasi tidak dapat berjalan."));
  }
}

class ErrorApp extends StatelessWidget {
  final String errorMessage;
  const ErrorApp(this.errorMessage, {super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: Scaffold(
        appBar: AppBar(title: const Text("Error"), backgroundColor: Colors.red),
        body: Center(
          child: Padding(
            padding: const EdgeInsets.all(16.0),
            child: Text(
              errorMessage,
              textAlign: TextAlign.center,
              style: const TextStyle(fontSize: 16),
            ),
          ),
        ),
      ),
    );
  }
}

class MyApp extends StatelessWidget {
  final CameraDescription camera;
  const MyApp({super.key, required this.camera});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Deteksi Fraksi Sawit',
      theme: ThemeData.dark(),
      home: ObjectDetectionView(camera: camera),
    );
  }
}

class ObjectDetectionView extends StatefulWidget {
  final CameraDescription camera;
  const ObjectDetectionView({super.key, required this.camera});

  @override
  State<ObjectDetectionView> createState() => _ObjectDetectionViewState();
}

class _ObjectDetectionViewState extends State<ObjectDetectionView> {
  CameraController? _cameraController;
  Interpreter? _interpreter;
  late List<String> _labels;
  bool _isDetecting = false;
  List<Map<String, dynamic>> _recognitions = [];
  late Future<void> _initializeControllerFuture;
  final int _modelInputSize = 640;
  int frameCounter = 0;
  final int frameSkipRate = 15;

  @override
  void initState() {
    super.initState();
    _initializeControllerFuture = _loadModelAndCamera();
  }

  Future<void> _loadModelAndCamera() async {
    _interpreter = await Interpreter.fromAsset('assets/best_int8.tflite');
    if (!mounted) return;
    final labelsData = await DefaultAssetBundle.of(context).loadString('assets/labels.txt');
    _labels = labelsData.split('\n').where((label) => label.isNotEmpty).toList();
    _cameraController = CameraController(widget.camera, ResolutionPreset.low, enableAudio: false);
    await _cameraController!.initialize();
    if (mounted) {
      _cameraController?.startImageStream((CameraImage image) {
        frameCounter++;
        if (frameCounter % frameSkipRate == 0) {
          if (!_isDetecting) {
            _isDetecting = true;
            _runModelOnFrame(image);
          }
          frameCounter = 0;
        }
      });
    }
  }

  @override
  void dispose() {
    _cameraController?.dispose();
    _interpreter?.close();
    super.dispose();
  }

  Future<void> _runModelOnFrame(CameraImage cameraImage) async {
    if (_interpreter == null) return;
    final recognitions = await compute(
      runModelOnIsolate,
      IsolateData(
        cameraImage,
        _interpreter!.address,
        _labels,
        _modelInputSize,
      ),
    );
    if (mounted) {
      setState(() {
        _recognitions = recognitions;
      });
    }
    _isDetecting = false;
  }

  @override
  Widget build(BuildContext context) {
    return FutureBuilder<void>(
      future: _initializeControllerFuture,
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.done) {
          if (snapshot.hasError) {
            return ErrorApp("Gagal menginisialisasi: ${snapshot.error}");
          }
          return Scaffold(
            appBar: AppBar(title: const Text('Deteksi Fraksi Sawit')),
            body: Stack(
              fit: StackFit.expand,
              children: [
                CameraPreview(_cameraController!),
                CustomPaint(
                  painter: BoundingBoxPainter(
                    recognitions: _recognitions,
                    modelInputSize: _modelInputSize,
                  ),
                ),
              ],
            ),
          );
        } else {
          return const Scaffold(
            body: Center(child: CircularProgressIndicator()),
          );
        }
      },
    );
  }
}

class BoundingBoxPainter extends CustomPainter {
  final List<Map<String, dynamic>> recognitions;
  final int modelInputSize;

  BoundingBoxPainter({required this.recognitions, required this.modelInputSize});

  @override
  void paint(Canvas canvas, Size size) {
    if (recognitions.isEmpty) return;
    final double scaleX = size.width / modelInputSize;
    final double scaleY = size.height / modelInputSize;

    for (var rec in recognitions) {
      final rect = rec['rect'] as Rect;
      final scaledRect = Rect.fromLTRB(
        rect.left * scaleX,
        rect.top * scaleY,
        rect.right * scaleX,
        rect.bottom * scaleY,
      );
      final paint = Paint()
        ..style = PaintingStyle.stroke
        ..strokeWidth = 2.0
        ..color = Colors.lightGreenAccent;
      canvas.drawRect(scaledRect, paint);
      final textPainter = TextPainter(
        text: TextSpan(
          text: '${rec['label']} ${(rec['score'] * 100).toStringAsFixed(0)}%',
          style: const TextStyle(
            color: Colors.white,
            backgroundColor: Colors.black54,
            fontSize: 12,
          ),
        ),
        textAlign: TextAlign.left,
        textDirection: TextDirection.ltr,
      );
      textPainter.layout();
      textPainter.paint(canvas, Offset(scaledRect.left, scaledRect.top - textPainter.height));
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) {
    return true;
  }
}