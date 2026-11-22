package org.bharat.edgeinspect;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.RectF;
import android.os.SystemClock;

import java.io.File;
import java.io.InputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.StandardCopyOption;
import org.json.JSONObject;
import org.pytorch.executorch.EValue;
import org.pytorch.executorch.Module;
import org.pytorch.executorch.Tensor;

final class VisionEngine implements AutoCloseable {
    static final String[] LABELS = {"Good", "Scratch", "Bent", "Discoloration", "Flipped"};
    private static final int INPUT_SIZE = 192;
    private static final int MASK_SIZE = 48;
    private final Module module;
    private final double classThreshold;
    private final double maskThreshold;
    private final double anomalyThreshold;

    record Finding(String decision, String defect, double confidence, String severity,
                   double area, RectF box, double milliseconds) {}

    VisionEngine(Context context) throws Exception {
        try (InputStream stream = context.getAssets().open("vision.json")) {
            JSONObject settings = new JSONObject(new String(stream.readAllBytes(), java.nio.charset.StandardCharsets.UTF_8));
            classThreshold = settings.getDouble("class_threshold");
            maskThreshold = settings.getDouble("mask_threshold");
            anomalyThreshold = settings.getDouble("anomaly_review_threshold");
        }
        File model = new File(context.getCacheDir(), "vision.pte");
        try (InputStream stream = context.getAssets().open("vision.pte")) {
            Files.copy(stream, model.toPath(), StandardCopyOption.REPLACE_EXISTING);
        }
        module = Module.load(model.getPath(), Module.LOAD_MODE_MMAP, 2);
    }

    Finding inspect(Bitmap image, boolean checkFraming) {
        long start = SystemClock.elapsedRealtimeNanos();
        Bitmap resized = Bitmap.createScaledBitmap(image, INPUT_SIZE, INPUT_SIZE, true);
        int plane = INPUT_SIZE * INPUT_SIZE;
        int[] pixels = new int[plane];
        resized.getPixels(pixels, 0, INPUT_SIZE, 0, 0, INPUT_SIZE, INPUT_SIZE);
        if (resized != image) resized.recycle();

        float[] rgb = new float[3 * plane];
        double sum = 0;
        double sumSquared = 0;
        for (int index = 0; index < plane; index++) {
            int color = pixels[index];
            float red = ((color >> 16) & 255) / 255f;
            float green = ((color >> 8) & 255) / 255f;
            float blue = (color & 255) / 255f;
            rgb[index] = red;
            rgb[plane + index] = green;
            rgb[2 * plane + index] = blue;
            double luma = 0.299 * red + 0.587 * green + 0.114 * blue;
            sum += luma;
            sumSquared += luma * luma;
        }
        double mean = sum / plane;
        double variance = sumSquared / plane - mean * mean;
        if (checkFraming && (mean < 0.06 || mean > 0.94 || variance < 0.0025)) {
            return new Finding("Review", "Image quality", 0, "Uncertain", 0, null, elapsed(start));
        }

        EValue[] outputs = module.forward(EValue.from(Tensor.fromBlob(
                rgb, new long[]{1, 3, INPUT_SIZE, INPUT_SIZE})));
        float[] logits = outputs[0].toTensor().getDataAsFloatArray();
        float[] mask = outputs[1].toTensor().getDataAsFloatArray();
        int best = 0;
        for (int index = 1; index < logits.length; index++) {
            if (logits[index] > logits[best]) best = index;
        }
        double denominator = 0;
        for (float logit : logits) denominator += Math.exp(logit - logits[best]);
        double confidence = 1 / denominator;
        if (!Double.isFinite(confidence)) {
            throw new IllegalStateException("Invalid scores from the vision model");
        }

        Region region = largestRegion(mask);
        double fraction = region.pixels / (double) (MASK_SIZE * MASK_SIZE);
        boolean anomaly = best == 0 && fraction >= anomalyThreshold;
        String defect = anomaly ? "Surface anomaly" : LABELS[best];
        String decision = confidence < classThreshold || anomaly ? "Review" : best == 0 ? "Pass" : "Hold";
        RectF box = best != 0 || anomaly ? region.box : null;
        String severity = "Review".equals(decision) ? "Uncertain"
                : best == 0 ? "None"
                : box == null ? "Uncertain"
                : fraction >= 0.05 ? "High"
                : fraction >= 0.01 ? "Medium" : "Low";
        return new Finding(decision, defect, confidence, severity, box == null ? 0 : fraction, box, elapsed(start));
    }

    private record Region(RectF box, int pixels) {}

    private Region largestRegion(float[] mask) {
        // Keep one contiguous region so isolated mask noise cannot inflate the displayed area.
        double minimumLogit = Math.log(maskThreshold / (1.0 - maskThreshold));
        boolean[] seen = new boolean[mask.length];
        int[] queue = new int[mask.length];
        int largest = 2;
        RectF box = null;
        for (int origin = 0; origin < mask.length; origin++) {
            if (seen[origin] || mask[origin] < minimumLogit) continue;
            seen[origin] = true;
            queue[0] = origin;
            int tail = 1;
            int left = MASK_SIZE, top = MASK_SIZE, right = 0, bottom = 0;
            for (int head = 0; head < tail; head++) {
                int cell = queue[head];
                int x = cell % MASK_SIZE, y = cell / MASK_SIZE;
                left = Math.min(left, x);
                top = Math.min(top, y);
                right = Math.max(right, x + 1);
                bottom = Math.max(bottom, y + 1);
                if (x > 0) tail = add(mask, minimumLogit, seen, queue, tail, cell - 1);
                if (x < MASK_SIZE - 1) tail = add(mask, minimumLogit, seen, queue, tail, cell + 1);
                if (y > 0) tail = add(mask, minimumLogit, seen, queue, tail, cell - MASK_SIZE);
                if (y < MASK_SIZE - 1) tail = add(mask, minimumLogit, seen, queue, tail, cell + MASK_SIZE);
            }
            if (tail > largest) {
                largest = tail;
                box = new RectF(left / (float) MASK_SIZE, top / (float) MASK_SIZE,
                        right / (float) MASK_SIZE, bottom / (float) MASK_SIZE);
            }
        }
        return new Region(box, box == null ? 0 : largest);
    }

    private static int add(float[] mask, double threshold, boolean[] seen, int[] queue, int tail, int cell) {
        if (!seen[cell] && mask[cell] >= threshold) {
            seen[cell] = true;
            queue[tail++] = cell;
        }
        return tail;
    }

    static double elapsed(long start) {
        return (SystemClock.elapsedRealtimeNanos() - start) / 1_000_000.0;
    }

    @Override public void close() { module.close(); }
}
