package org.bharat.edgeinspect;

import android.Manifest;
import android.app.AlertDialog;
import android.content.Intent;
import android.content.pm.ApplicationInfo;
import android.content.pm.PackageManager;
import android.content.res.ColorStateList;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageDecoder;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.graphics.drawable.GradientDrawable;
import android.net.Uri;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.util.Size;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import androidx.activity.ComponentActivity;
import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.camera.core.CameraSelector;
import androidx.camera.core.ImageAnalysis;
import androidx.camera.core.ImageProxy;
import androidx.camera.lifecycle.ProcessCameraProvider;
import androidx.core.content.ContextCompat;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import java.security.DigestInputStream;
import java.security.MessageDigest;
import java.util.Arrays;
import java.util.Locale;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class MainActivity extends ComponentActivity {
    private static final String TAG = "EdgeInspect";
    private static final int BACKGROUND = 0xff101820;
    private static final int CARD = 0xff1b2833;
    private static final int MUTED = 0xffb5c3cf;
    private static final int MINT = 0xff85ded0;
    private static final int AMBER = 0xffffc775;
    private static final int CORAL = 0xffff9a8d;

    private final ExecutorService visionWorker = Executors.newSingleThreadExecutor();
    private final ExecutorService advisoryWorker = Executors.newSingleThreadExecutor();
    private final ExecutorService storageWorker = Executors.newSingleThreadExecutor();
    private final ActivityResultLauncher<String> cameraPermission = registerForActivityResult(
            new ActivityResultContracts.RequestPermission(), granted -> {
                if (granted) openCamera();
                else this.status.setText("Camera permission was declined. You can inspect an image instead.");
            });
    private final ActivityResultLauncher<String[]> imagePicker = registerForActivityResult(
            new ActivityResultContracts.OpenDocument(), uri -> {
                if (uri != null) openImage(uri);
            });
    private final ActivityResultLauncher<String[]> modelPicker = registerForActivityResult(
            new ActivityResultContracts.OpenDocument(), uri -> {
                if (uri != null) installModel(uri, this.modelFileToImport);
            });
    private final ActivityResultLauncher<String> csvPicker = registerForActivityResult(
            new ActivityResultContracts.CreateDocument("text/csv"), uri -> {
                if (uri != null) exportLog(uri);
            });

    private static final String CAMERA = "Live camera";
    private static final String IMAGE = "Imported image";

    private TextView status;
    private TextView decision;
    private TextView details;
    private TextView adviceText;
    private TextView metrics;
    private Button inspectButton;
    private FrameView frameView;
    private InspectionLog log;
    private VisionEngine vision;
    private AdvisoryEngine advisor;
    private ProcessCameraProvider camera;
    private Observation latest;
    private volatile int modeVersion;
    private String modelFileToImport;
    private long selectedInspection;
    private boolean closed;

    private record Observation(Bitmap bitmap, VisionEngine.Finding finding, String source, long capturedAt) {}

    @Override public void onCreate(Bundle state) {
        super.onCreate(state);
        log = new InspectionLog(getApplicationContext());
        advisor = new AdvisoryEngine(getFilesDir());
        if (state != null) modelFileToImport = state.getString("model_file");
        buildScreen();
        if (isDebuggable() && handleDebug(getIntent())) return;
        status.setText("Choose the live camera or an image to begin.");
    }

    @Override protected void onSaveInstanceState(Bundle state) {
        super.onSaveInstanceState(state);
        state.putString("model_file", modelFileToImport);
    }

    @Override protected void onNewIntent(Intent intent) {
        super.onNewIntent(intent);
        setIntent(intent);
        if (isDebuggable()) handleDebug(intent);
    }

    private boolean isDebuggable() {
        return (getApplicationInfo().flags & ApplicationInfo.FLAG_DEBUGGABLE) != 0;
    }

    private void buildScreen() {
        ScrollView scroll = new ScrollView(this);
        scroll.setFillViewport(true);
        scroll.setBackgroundColor(BACKGROUND);
        LinearLayout page = column();
        page.setPadding(dp(20), dp(18), dp(20), dp(20));
        scroll.addView(page);
        setContentView(scroll);

        LinearLayout header = new LinearLayout(this);
        TextView title = text("EdgeInspect", 29, Color.WHITE, true);
        header.addView(title, new LinearLayout.LayoutParams(0, -2, 1));
        Button model = button("Models");
        model.setOnClickListener(view -> modelSetup());
        header.addView(model);
        page.addView(header);
        add(page, text("Metal nut visual inspection", 15, MUTED, false), 4);
        add(page, text("On-device · Arm CPU", 12, MINT, true), 6);

        LinearLayout modes = new LinearLayout(this);
        Button cameraButton = button("Live camera");
        cameraButton.setOnClickListener(view -> startCamera());
        Button imageButton = button("Import image");
        imageButton.setOnClickListener(view -> imagePicker.launch(new String[]{"image/*"}));
        modes.addView(cameraButton, new LinearLayout.LayoutParams(0, dp(48), 1));
        LinearLayout.LayoutParams second = new LinearLayout.LayoutParams(0, dp(48), 1);
        second.leftMargin = dp(8);
        modes.addView(imageButton, second);
        add(page, modes, 12);

        frameView = new FrameView();
        LinearLayout.LayoutParams frameLayout = new LinearLayout.LayoutParams(-1,
                dp(Math.min(260, getResources().getConfiguration().screenWidthDp - 40)));
        frameLayout.topMargin = dp(10);
        page.addView(frameView, frameLayout);
        status = text("", 13, MUTED, false);
        add(page, status, 8);

        LinearLayout panel = column();
        panel.setPadding(dp(14), dp(12), dp(14), dp(12));
        panel.setBackground(shape(CARD, dp(12)));
        decision = text("Awaiting a part", 21, Color.WHITE, true);
        panel.addView(decision);
        details = text("", 14, MUTED, false);
        add(panel, details, 5);
        adviceText = text("", 14, Color.WHITE, false);
        add(panel, adviceText, 7);
        metrics = text("", 12, MUTED, false);
        add(panel, metrics, 7);

        inspectButton = button("Inspect part");
        inspectButton.setEnabled(false);
        inspectButton.setBackgroundTintList(ColorStateList.valueOf(MINT));
        inspectButton.setTextColor(BACKGROUND);
        inspectButton.setOnClickListener(view -> inspectCurrent());
        LinearLayout actions = new LinearLayout(this);
        actions.addView(inspectButton, new LinearLayout.LayoutParams(0, dp(48), 1));

        Button history = button("Inspection history");
        history.setOnClickListener(view -> openHistory());
        LinearLayout.LayoutParams historyLayout = new LinearLayout.LayoutParams(0, dp(48), 1);
        historyLayout.leftMargin = dp(8);
        actions.addView(history, historyLayout);
        add(page, actions, 8);
        add(page, panel, 8);
    }

    private void startCamera() {
        if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
            openCamera();
        } else {
            cameraPermission.launch(Manifest.permission.CAMERA);
        }
    }

    private void openCamera() {
        int version = switchMode("Opening the rear camera…");
        getWindow().addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        var future = ProcessCameraProvider.getInstance(this);
        future.addListener(() -> {
            if (closed || version != modeVersion) return;
            try {
                camera = future.get();
                camera.unbindAll();
                ImageAnalysis analyzer = new ImageAnalysis.Builder()
                        .setTargetResolution(new Size(640, 480))
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .build();
                analyzer.setAnalyzer(visionWorker, frame -> analyzeCameraFrame(frame, version));
                camera.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, analyzer);
            } catch (Exception failure) {
                showFailure("Camera unavailable. Import an image to continue.", failure);
            }
        }, ContextCompat.getMainExecutor(this));
    }

    private int switchMode(String message) {
        modeVersion++;
        getWindow().clearFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        if (camera != null) camera.unbindAll();
        latest = null;
        selectedInspection = 0;
        lastCameraFrame = 0;
        cameraTimings.reset();
        inspectButton.setEnabled(false);
        frameView.show(null);
        status.setText(message);
        decision.setText("Awaiting a part");
        decision.setTextColor(Color.WHITE);
        details.setText("");
        adviceText.setText("");
        metrics.setText("");
        return modeVersion;
    }

    private volatile long lastCameraFrame;
    private final CameraTimings cameraTimings = new CameraTimings();

    private void analyzeCameraFrame(ImageProxy frame, int version) {
        long now = SystemClock.elapsedRealtime();
        try {
            if (version != modeVersion || now - lastCameraFrame < 60) return;
            lastCameraFrame = now;
            Bitmap bitmap = frame.toBitmap();
            try {
                Bitmap square = square(bitmap, frame.getImageInfo().getRotationDegrees());
                Observation observation = analyze(square, CAMERA);
                String performance = observation.finding.confidence() > 0
                        ? cameraTimings.add(observation.finding.milliseconds()) : "Adjust the lighting or framing.";
                runOnUiThread(() -> {
                    if (closed || version != modeVersion) return;
                    showObservation(observation);
                    metrics.setText(performance);
                });
            } finally {
                bitmap.recycle();
            }
        } catch (Exception | LinkageError failure) {
            runOnUiThread(() -> {
                if (version == modeVersion && !closed) {
                    switchMode("Camera analysis failed.");
                    showFailure("Camera analysis failed. Try importing an image.", failure);
                }
            });
        } finally {
            frame.close();
        }
    }

    private void openImage(Uri uri) {
        int version = switchMode("Analyzing the selected image…");
        visionWorker.execute(() -> {
            try {
                ImageDecoder.Source source = ImageDecoder.createSource(getContentResolver(), uri);
                Bitmap image = ImageDecoder.decodeBitmap(source, (decoder, info, unused) -> {
                    decoder.setAllocator(ImageDecoder.ALLOCATOR_SOFTWARE);
                    int largest = Math.max(info.getSize().getWidth(), info.getSize().getHeight());
                    decoder.setTargetSampleSize(Math.max(1, largest / 1200));
                });
                submitImage(image, version, false, 0);
            } catch (Exception | LinkageError failure) {
                runOnUiThread(() -> {
                    if (version == modeVersion && !closed) showFailure("Image analysis failed.", failure);
                });
            }
        });
    }

    private void submitImage(Bitmap bitmap, int version, boolean inspect, int benchmark) throws Exception {
        Bitmap square;
        try {
            square = square(bitmap, 0);
        } finally {
            bitmap.recycle();
        }
        Observation observation = analyze(square, IMAGE);
        if (benchmark > 0) runBenchmark(square, benchmark);
        runOnUiThread(() -> {
            if (closed || version != modeVersion) return;
            showObservation(observation);
            metrics.setText(String.format(Locale.US, "Vision %.1f ms", observation.finding.milliseconds()));
            if (inspect) inspectCurrent();
        });
    }

    private Observation analyze(Bitmap image, String source) throws Exception {
        if (vision == null) vision = new VisionEngine(getApplicationContext());
        return new Observation(image, vision.inspect(image, CAMERA.equals(source)), source,
                SystemClock.elapsedRealtime());
    }

    private void showObservation(Observation observation) {
        latest = observation;
        inspectButton.setEnabled(true);
        frameView.show(observation);
        if (selectedInspection != 0) return;
        status.setText(CAMERA.equals(observation.source) ? "Live camera · center one part in view"
                : "Imported image · ready to inspect");
        showFinding(observation.finding);
    }

    private void showFinding(VisionEngine.Finding finding) {
        decision.setText(finding.decision());
        decision.setTextColor(switch (finding.decision()) {
            case "Hold" -> CORAL;
            case "Review" -> AMBER;
            default -> MINT;
        });
        details.setText(String.format(Locale.US, "%s  ·  %.0f%% confidence  ·  %s severity",
                finding.defect(), finding.confidence() * 100, finding.severity()));
    }

    private void inspectCurrent() {
        Observation observation = latest;
        if (observation == null) return;
        if (CAMERA.equals(observation.source) && SystemClock.elapsedRealtime() - observation.capturedAt > 1500) {
            status.setText("Waiting for a fresh camera frame.");
            return;
        }
        AdvisoryEngine.Advice fallback = AdvisoryEngine.fallback(observation.finding);
        long id;
        try {
            id = log.save(observation.source, observation.finding, fallback);
        } catch (RuntimeException failure) {
            showFailure("Inspection could not be saved.", failure);
            return;
        }
        selectedInspection = id;
        frameView.show(observation);
        showFinding(observation.finding);
        Log.i(TAG, String.format(Locale.US,
                "Saved inspection id=%d source=%s decision=%s defect=%s confidence=%.4f severity=%s box=%s vision_ms=%.2f",
                id, observation.source, observation.finding.decision(), observation.finding.defect(),
                observation.finding.confidence(), observation.finding.severity(),
                observation.finding.box(), observation.finding.milliseconds()));
        status.setText("Inspection #" + id + " saved · " + observation.source);
        showAdvice(fallback);
        if (!advisor.hasFiles() || fallback.source().isEmpty()) return;
        adviceText.setText(fallback.action() + "\nGenerating local explanation…");
        advisoryWorker.execute(() -> {
            AdvisoryEngine.Advice advice = advisor.recommend(observation.finding);
            try {
                log.updateAdvice(id, advice);
            } catch (RuntimeException failure) {
                runOnUiThread(() -> {
                    if (!closed && selectedInspection == id) {
                        showAdvice(fallback);
                        showFailure("Could not update the saved advisory; default guidance is available.", failure);
                    }
                });
                return;
            }
            Log.i(TAG, String.format(Locale.US, "Advice id=%d source=%s first_token_ms=%.2f action=%s",
                    id, advice.source(), advice.firstTokenMs(), advice.action()));
            runOnUiThread(() -> {
                if (!closed && selectedInspection == id) showAdvice(advice);
            });
        });
    }

    private void showAdvice(AdvisoryEngine.Advice advice) {
        if (advice.source().isEmpty()) {
            adviceText.setText("");
            return;
        }
        adviceText.setText(advice.action() + "\n" + advice.explanation() + "\n" + advice.source());
    }

    private void modelSetup() {
        new AlertDialog.Builder(this)
                .setTitle(advisor.hasFiles() ? "Local Llama files installed" : "Set up local Llama")
                .setItems(new String[]{"Import Llama .pte", "Import tokenizer.json", "Run local Llama check"},
                        (dialog, choice) -> {
                            if (choice == 2) runSmoke();
                            else {
                                modelFileToImport = choice == 0 ? "llama.pte" : "tokenizer.json";
                                modelPicker.launch(new String[]{"*/*"});
                            }
                        })
                .setNegativeButton("Close", null).show();
    }

    private void installModel(Uri uri, String name) {
        if (!"llama.pte".equals(name) && !"tokenizer.json".equals(name)) return;
        status.setText("Copying " + name + " into local storage…");
        advisoryWorker.execute(() -> {
            File pending = new File(getFilesDir(), name + ".partial");
            try (InputStream input = getContentResolver().openInputStream(uri)) {
                if (input == null) throw new IOException("Selected file could not be opened");
                boolean isModel = "llama.pte".equals(name);
                long expectedSize = isModel ? AdvisoryEngine.MODEL_SIZE : AdvisoryEngine.TOKENIZER_SIZE;
                String expectedHash = isModel ? AdvisoryEngine.MODEL_SHA256 : AdvisoryEngine.TOKENIZER_SHA256;
                MessageDigest hash = MessageDigest.getInstance("SHA-256");
                try (DigestInputStream verified = new DigestInputStream(input, hash)) {
                    Files.copy(verified, pending.toPath(), StandardCopyOption.REPLACE_EXISTING);
                }
                String actualHash = String.format(Locale.ROOT, "%064x", new java.math.BigInteger(1, hash.digest()));
                if (pending.length() != expectedSize || !expectedHash.equals(actualHash)) {
                    throw new IOException("File does not match the pinned Llama download");
                }
                advisor.close();
                Files.move(pending.toPath(), new File(getFilesDir(), name).toPath(),
                        StandardCopyOption.REPLACE_EXISTING);
                runOnUiThread(() -> {
                    if (!closed) status.setText(name + " installed locally.");
                });
            } catch (Exception failure) {
                runOnUiThread(() -> {
                    if (!closed) {
                        showFailure("Could not install " + name + ". Choose the exact file from the setup guide.", failure);
                    }
                });
            } finally {
                if (pending.exists() && !pending.delete()) Log.w(TAG, "Could not remove temporary model file");
            }
        });
    }

    private void runSmoke() {
        status.setText("Checking Llama locally…");
        advisoryWorker.execute(() -> {
            try {
                String reply = advisor.smoke();
                if (reply.isBlank()) throw new IllegalStateException("Llama produced an empty reply");
                runOnUiThread(() -> {
                    if (!closed) {
                        status.setText("Local Llama check passed.");
                        adviceText.setText(reply.trim().split("<\\|", 2)[0]);
                    }
                });
            } catch (Exception | LinkageError failure) {
                runOnUiThread(() -> {
                    if (!closed) showFailure("Local Llama check failed.", failure);
                });
            }
        });
    }

    private void openHistory() {
        storageWorker.execute(() -> {
            try {
                var entries = log.latest();
                runOnUiThread(() -> {
                    if (closed) return;
                    ScrollView scroll = new ScrollView(this);
                    LinearLayout list = column();
                    list.setPadding(dp(20), dp(8), dp(20), dp(8));
                    scroll.addView(list);
                    if (entries.isEmpty()) list.addView(text("No inspections recorded yet.", 14, MUTED, false));
                    for (InspectionLog.Entry entry : entries) {
                        add(list, text("#" + entry.id() + "   " + entry.detail(), 15, Color.WHITE, true), 12);
                        add(list, text(entry.timestamp(), 12, MUTED, false), 3);
                        if (!entry.advice().isEmpty()) {
                            add(list, text(entry.advice() + "\n" + entry.adviceSource(), 13, MINT, false), 4);
                        }
                    }
                    new AlertDialog.Builder(this).setTitle("Inspection history").setView(scroll)
                            .setPositiveButton("Export CSV", (dialog, choice) -> csvPicker.launch("edge-inspections.csv"))
                            .setNegativeButton("Close", null).show();
                });
            } catch (RuntimeException failure) {
                runOnUiThread(() -> {
                    if (!closed) showFailure("Could not open inspection history.", failure);
                });
            }
        });
    }

    private void exportLog(Uri uri) {
        storageWorker.execute(() -> {
            try (var stream = getContentResolver().openOutputStream(uri);
                 var writer = new OutputStreamWriter(stream, StandardCharsets.UTF_8)) {
                log.export(writer);
                runOnUiThread(() -> {
                    if (!closed) status.setText("Inspection CSV exported.");
                });
            } catch (Exception failure) {
                runOnUiThread(() -> {
                    if (!closed) showFailure("Could not export the CSV.", failure);
                });
            }
        });
    }

    private boolean handleDebug(Intent intent) {
        if (intent.getBooleanExtra("smoke", false)) {
            runSmoke();
            return true;
        }
        if (intent.getBooleanExtra("evaluate", false)) {
            switchMode("Evaluating held-out images locally…");
            visionWorker.execute(this::evaluateFixtures);
            return true;
        }
        if (intent.getBooleanExtra("camera", false)) {
            startCamera();
            return true;
        }
        String fixture = intent.getStringExtra("fixture");
        if (fixture == null || !fixture.matches("[A-Za-z0-9_.-]+\\.(jpg|png)")) return false;
        int version = switchMode("Analyzing a local validation image…");
        visionWorker.execute(() -> {
            try {
                File input = new File(getFilesDir(), fixture);
                Bitmap image = ImageDecoder.decodeBitmap(ImageDecoder.createSource(input),
                        (decoder, info, unused) -> decoder.setAllocator(ImageDecoder.ALLOCATOR_SOFTWARE));
                submitImage(image, version, intent.getBooleanExtra("inspect", false),
                        Math.min(1000, intent.getIntExtra("benchmark", 0)));
            } catch (Exception | LinkageError failure) {
                runOnUiThread(() -> {
                    if (!closed) showFailure("Validation image failed.", failure);
                });
            }
        });
        return true;
    }

    private void evaluateFixtures() {
        File[] fixtures = new File(getFilesDir(), "validation").listFiles(
                (directory, name) -> name.endsWith(".png"));
        if (fixtures == null || fixtures.length == 0) {
            runOnUiThread(() -> status.setText("No local validation images."));
            return;
        }
        Arrays.sort(fixtures);
        try {
            for (File file : fixtures) {
                Bitmap input = ImageDecoder.decodeBitmap(ImageDecoder.createSource(file),
                        (decoder, info, unused) -> decoder.setAllocator(ImageDecoder.ALLOCATOR_SOFTWARE));
                Bitmap prepared = square(input, 0);
                input.recycle();
                Observation observation = analyze(prepared, IMAGE);
                prepared.recycle();
                VisionEngine.Finding result = observation.finding;
                Log.i(TAG, String.format(Locale.US, "Validation sample=%s decision=%s class=%s confidence=%.4f vision_ms=%.2f",
                        file.getName(), result.decision(), result.defect(), result.confidence(), result.milliseconds()));
            }
            runOnUiThread(() -> status.setText("Validated " + fixtures.length + " local images."));
        } catch (Exception | LinkageError failure) {
            runOnUiThread(() -> showFailure("Validation run failed.", failure));
        }
    }

    private void runBenchmark(Bitmap sample, int frames) {
        for (int count = 0; count < 10; count++) vision.inspect(sample, false);
        double[] elapsed = new double[frames];
        for (int count = 0; count < frames; count++) elapsed[count] = vision.inspect(sample, false).milliseconds();
        Arrays.sort(elapsed);
        Log.i(TAG, String.format(Locale.US, "Vision benchmark mode=imported frames=%d median_ms=%.2f p95_ms=%.2f",
                frames, elapsed[(frames - 1) / 2], elapsed[(int) Math.ceil(frames * .95) - 1]));
    }

    private void showFailure(String message, Throwable error) {
        Log.e(TAG, message, error);
        status.setText(message);
    }

    @Override protected void onDestroy() {
        closed = true;
        modeVersion++;
        if (camera != null) camera.unbindAll();
        visionWorker.execute(() -> {
            if (vision != null) vision.close();
        });
        advisoryWorker.execute(() -> {
            advisor.close();
            storageWorker.execute(log::close);
            storageWorker.shutdown();
        });
        visionWorker.shutdown();
        advisoryWorker.shutdown();
        super.onDestroy();
    }

    private Bitmap square(Bitmap bitmap, int rotation) {
        int side = Math.min(bitmap.getWidth(), bitmap.getHeight());
        Bitmap scaled = Bitmap.createBitmap(384, 384, Bitmap.Config.ARGB_8888);
        Canvas canvas = new Canvas(scaled);
        Matrix matrix = new Matrix();
        matrix.postTranslate(-bitmap.getWidth() / 2f, -bitmap.getHeight() / 2f);
        matrix.postRotate(rotation);
        matrix.postScale(384f / side, 384f / side);
        matrix.postTranslate(192, 192);
        canvas.drawBitmap(bitmap, matrix, new Paint(Paint.FILTER_BITMAP_FLAG));
        return scaled;
    }

    private final class FrameView extends View {
        private final Paint paint = new Paint(Paint.ANTI_ALIAS_FLAG | Paint.FILTER_BITMAP_FLAG);
        private Observation observation;

        FrameView() {
            super(MainActivity.this);
            setBackground(shape(CARD, dp(12)));
            setClipToOutline(true);
            setContentDescription("Inspection image and defect location");
        }

        void show(Observation next) {
            observation = next;
            invalidate();
        }

        @Override protected void onDraw(Canvas canvas) {
            super.onDraw(canvas);
            float side = Math.min(getWidth(), getHeight());
            float left = (getWidth() - side) / 2, top = (getHeight() - side) / 2;
            if (observation == null) {
                paint.setStyle(Paint.Style.FILL);
                paint.setColor(MUTED);
                paint.setTextAlign(Paint.Align.CENTER);
                paint.setTextSize(dp(14));
                canvas.drawText("Camera or imported image", getWidth() / 2f, getHeight() / 2f, paint);
                return;
            }
            paint.setStyle(Paint.Style.FILL);
            canvas.drawBitmap(observation.bitmap, null, new RectF(left, top, left + side, top + side), paint);
            RectF rectangle = observation.finding.box();
            if (rectangle == null) return;
            paint.setStyle(Paint.Style.STROKE);
            paint.setStrokeWidth(dp(3));
            paint.setColor("Review".equals(observation.finding.decision()) ? AMBER : CORAL);
            canvas.drawRect(left + side * rectangle.left, top + side * rectangle.top,
                    left + side * rectangle.right, top + side * rectangle.bottom, paint);
            paint.setStyle(Paint.Style.FILL);
        }
    }

    private static final class CameraTimings {
        private final double[] values = new double[100];
        private int count;
        private int lastReport;
        private long start;

        synchronized void reset() {
            count = 0;
            lastReport = 0;
            start = 0;
        }

        synchronized String add(double ms) {
            long now = SystemClock.elapsedRealtime();
            if (count == 10) start = now;
            if (count >= 10) values[(count - 10) % values.length] = ms;
            count++;
            int samples = Math.min(values.length, Math.max(0, count - 10));
            double fps = count <= 11 ? 0 : (count - 11) * 1000.0 / Math.max(1, now - start);
            if (samples == values.length && count - lastReport >= 100) {
                double[] sorted = values.clone();
                Arrays.sort(sorted);
                Log.i(TAG, String.format(Locale.US, "Camera benchmark frames=100 fps=%.1f median_ms=%.2f p95_ms=%.2f",
                        fps, sorted[49], sorted[94]));
                lastReport = count;
            }
            return String.format(Locale.US, "Vision %.1f ms  ·  %.1f analyzed FPS", ms, fps);
        }
    }

    private LinearLayout column() {
        LinearLayout body = new LinearLayout(this);
        body.setOrientation(LinearLayout.VERTICAL);
        return body;
    }

    private TextView text(String label, int size, int color, boolean bold) {
        TextView text = new TextView(this);
        text.setText(label);
        text.setTextSize(size);
        text.setTextColor(color);
        if (bold) text.setTypeface(Typeface.DEFAULT, Typeface.BOLD);
        return text;
    }

    private Button button(String label) {
        Button button = new Button(this);
        button.setText(label);
        button.setAllCaps(false);
        button.setTextColor(Color.WHITE);
        button.setBackgroundTintList(ColorStateList.valueOf(CARD));
        return button;
    }

    private void add(LinearLayout parent, View view, int top) {
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(-1, -2);
        params.topMargin = dp(top);
        parent.addView(view, params);
    }

    private GradientDrawable shape(int color, int radius) {
        GradientDrawable shape = new GradientDrawable();
        shape.setColor(color);
        shape.setCornerRadius(radius);
        return shape;
    }

    private int dp(int amount) {
        return (int) (getResources().getDisplayMetrics().density * amount + .5f);
    }
}
