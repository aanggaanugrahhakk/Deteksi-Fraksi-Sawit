-keep class org.tensorflow.lite.** { *; }
-keep interface org.tensorflow.lite.** { *; }
-keep class org.tensorflow.lite.gpu.GpuDelegateFactory$Options$* { *; }
-keep class org.tensorflow.lite.flex.** { *; }
-keep interface org.tensorflow.lite.flex.** { *; }
-keep class com.google.android.gms.tflite.gpu.** { *; }
-keep interface com.google.android.gms.tflite.gpu.** { *; }
-keep class com.google.auto.value.** { *; }
-keep @com.google.auto.value.AutoValue class * {
    <fields>;
    <methods>;
}