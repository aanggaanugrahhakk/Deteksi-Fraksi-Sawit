import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:camera/camera.dart';
import 'package:myapp/main.dart';

void main() {
  testWidgets('Counter increments smoke test', (WidgetTester tester) async {
    final mockCamera = const CameraDescription(
      name: 'mock_cam',
      lensDirection: CameraLensDirection.back,
      sensorOrientation: 90,
    );

    await tester.pumpWidget(MyApp(camera: mockCamera));

    expect(find.text('0'), findsOneWidget);
    expect(find.text('1'), findsNothing);

    await tester.tap(find.byIcon(Icons.add));
    await tester.pump();

    expect(find.text('0'), findsNothing);
    expect(find.text('1'), findsOneWidget);
  });
}
