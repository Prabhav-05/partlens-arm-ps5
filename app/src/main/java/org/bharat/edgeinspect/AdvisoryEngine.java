package org.bharat.edgeinspect;

import android.graphics.RectF;
import android.os.SystemClock;
import android.util.Log;

import java.io.File;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.pytorch.executorch.extension.llm.LlmCallback;
import org.pytorch.executorch.extension.llm.LlmGenerationConfig;
import org.pytorch.executorch.extension.llm.LlmModule;

final class AdvisoryEngine implements AutoCloseable {
    static final String TAG = "EdgeInspect";
    static final long MODEL_SIZE = 1_082_483_584L;
    static final long TOKENIZER_SIZE = 9_906_781L;
    static final String MODEL_SHA256 = "cd9b0bcc5a1d71072a818b9237dba45fc84fdf7b244fb455f63806bfc7111ad2";
    static final String TOKENIZER_SHA256 = "0b9897f5668a5d202662c4bab2be785eb987daf194b557315a690f3d2dff1ce0";
    private static final Pattern REPLY = Pattern.compile(
            "(?is)^\\s*ACTION\\s*:\\s*([A-Z_]+)\\s*[\\r\\n]+\\s*WHY\\s*:\\s*([^\\r\\n]{8,260})\\s*$");
    private static final Map<String, String> ACTIONS = Map.of(
            "REIMAGE", "Reposition the part under steady light and inspect it again.",
            "HOLD_FOR_QC", "Set the part aside for quality-control review.",
            "CHECK_GAUGE", "Set the part aside and ask QC to verify its dimensions.",
            "VERIFY_ORIENTATION", "Verify part orientation and rerun the inspection.");
    private final File directory;
    private LlmModule model;

    record Advice(String action, String explanation, String source, double firstTokenMs) {}

    AdvisoryEngine(File directory) {
        this.directory = directory;
    }

    boolean hasFiles() {
        return new File(directory, "llama.pte").length() == MODEL_SIZE
                && new File(directory, "tokenizer.json").length() == TOKENIZER_SIZE;
    }

    static Advice fallback(VisionEngine.Finding finding) {
        if ("Pass".equals(finding.decision())) return new Advice("", "", "", 0);
        if ("Review".equals(finding.decision())) {
            return new Advice(ACTIONS.get("REIMAGE"), "The image or prediction needs operator verification.",
                    "Default guidance", 0);
        }
        return new Advice(ACTIONS.get("HOLD_FOR_QC"), "The detector flagged " + finding.defect().toLowerCase(Locale.ROOT)
                + " for operator review.", "Default guidance", 0);
    }

    Advice recommend(VisionEngine.Finding finding) {
        Advice fallback = fallback(finding);
        if (!hasFiles() || fallback.source().isEmpty()) return fallback;
        String allowed = "Review".equals(finding.decision()) ? "REIMAGE"
                : switch (finding.defect()) {
                    case "Bent" -> "HOLD_FOR_QC, CHECK_GAUGE";
                    case "Flipped" -> "HOLD_FOR_QC, VERIFY_ORIENTATION";
                    default -> "HOLD_FOR_QC";
                };
        RectF box = finding.box();
        String location = box == null ? "unavailable"
                : String.format(Locale.US, "normalized box [%.2f,%.2f,%.2f,%.2f]", box.left, box.top, box.right, box.bottom);
        String facts = String.format(Locale.US, "Decision=%s; type=%s; severity=%s; confidence=%.0f%%; location=%s.",
                finding.decision(), finding.defect(), finding.severity(), finding.confidence() * 100, location);
        String prompt = "You assist a metal-nut quality inspector. Use only the inspection evidence. "
                + "Choose one allowed action. Explain the finding in one short sentence without extra instructions. "
                + "Reply on exactly two lines: ACTION: <allowed action ID> followed by WHY: <one short sentence>.\n"
                + "Allowed action IDs: " + allowed + ".\nInspection: " + facts;
        try {
            return parse(generate(prompt, 56), allowed, fallback);
        } catch (Exception | LinkageError failure) {
            Log.e(TAG, "Local advisory failed", failure);
            return fallback;
        }
    }

    String smoke() {
        return generate("A metal nut has a visible scratch. In one sentence, tell the inspector "
                + "to isolate the part for quality review.", 36).text;
    }

    private static Advice parse(Response response, String allowed, Advice fallback) {
        String output = response.text.split("<\\|", 2)[0].trim();
        Matcher match = REPLY.matcher(output);
        if (!match.matches()) return withTiming(fallback, response.firstTokenMs);
        String chosen = match.group(1).trim().toUpperCase(Locale.ROOT);
        if (!java.util.Arrays.asList(allowed.split(", ")).contains(chosen)) {
            return withTiming(fallback, response.firstTokenMs);
        }
        return new Advice(ACTIONS.get(chosen), match.group(2).trim(), "Llama / ExecuTorch", response.firstTokenMs);
    }

    private static Advice withTiming(Advice advice, double firstTokenMs) {
        return new Advice(advice.action, advice.explanation, advice.source, firstTokenMs);
    }

    private record Response(String text, double firstTokenMs) {}

    private Response generate(String input, int maxTokens) {
        if (!hasFiles()) throw new IllegalStateException("Install the Llama model and tokenizer first");
        if (model == null) {
            long start = SystemClock.elapsedRealtimeNanos();
            LlmModule candidate = new LlmModule(new File(directory, "llama.pte").getPath(),
                    new File(directory, "tokenizer.json").getPath(), 0f);
            try {
                candidate.load();
                model = candidate;
            } catch (RuntimeException | LinkageError failure) {
                candidate.close();
                throw failure;
            }
            Log.i(TAG, "Llama load_ms=" + VisionEngine.elapsed(start));
        }
        String prompt = "<|start_header_id|>user<|end_header_id|>\n\n" + input
                + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
        StringBuilder output = new StringBuilder();
        double[] firstToken = {0};
        String[] error = {null};
        long start = SystemClock.elapsedRealtimeNanos();
        model.generate(prompt, LlmGenerationConfig.create().numBos(1).numEos(0)
                .maxNewTokens(maxTokens).echo(false).temperature(0f).build(), new LlmCallback() {
            @Override public void onResult(String token) {
                if (firstToken[0] == 0 && !token.isBlank()) firstToken[0] = VisionEngine.elapsed(start);
                output.append(token);
            }
            @Override public void onError(int code, String message) { error[0] = code + ": " + message; }
            @Override public void onStats(String stats) { Log.i(TAG, "Llama stats " + stats); }
        });
        if (error[0] != null) throw new IllegalStateException(error[0]);
        Log.i(TAG, "Llama first_token_ms=" + firstToken[0] + " output=" + output);
        return new Response(output.toString(), firstToken[0]);
    }

    @Override public void close() {
        if (model != null) {
            model.close();
            model = null;
        }
    }
}
