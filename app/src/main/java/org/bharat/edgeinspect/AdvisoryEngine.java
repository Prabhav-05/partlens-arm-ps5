package org.bharat.edgeinspect;

import android.graphics.RectF;
import android.os.SystemClock;
import android.util.Log;

import java.io.File;
import java.util.List;
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
    private static final Pattern CHOICE = Pattern.compile("(?i)^\\s*([AB])(?:[.)]|[ \\t]*(?:\\R|$))");
    private static final Pattern UNSUPPORTED = Pattern.compile(
            "\\b(acceptable|accept|approve|release|rework|safe|confirmed|guaranteed|manufacturing cause|wear and tear)\\b");
    private static final Map<String, String> ACTIONS = Map.of(
            "REIMAGE", "Reposition the part under steady light and inspect it again.",
            "HOLD_FOR_QC", "Set the part aside for quality-control review.",
            "CHECK_GAUGE", "Set the part aside and ask QC to verify its dimensions.",
            "CHECK_SURFACE", "Set the part aside and ask QC to examine the marked surface.",
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
        List<String> allowed = "Review".equals(finding.decision()) ? List.of("REIMAGE", "HOLD_FOR_QC")
                : switch (finding.defect()) {
                    case "Bent" -> List.of("HOLD_FOR_QC", "CHECK_GAUGE");
                    case "Flipped" -> List.of("HOLD_FOR_QC", "VERIFY_ORIENTATION");
                    case "Scratch", "Discoloration" -> List.of("HOLD_FOR_QC", "CHECK_SURFACE");
                    default -> List.of("HOLD_FOR_QC");
                };
        RectF box = finding.box();
        String location = box == null ? "unavailable"
                : String.format(Locale.US, "normalized box [%.2f,%.2f,%.2f,%.2f]", box.left, box.top, box.right, box.bottom);
        String facts = String.format(Locale.US, "Decision=%s; type=%s; severity=%s; confidence=%.0f%%; location=%s.",
                finding.decision(), finding.defect(), finding.severity(), finding.confidence() * 100, location);
        String prompt = "A metal-nut inspector needs a next step. The vision model reports: " + facts
                + "\nSelect the best action. Confidence is certainty in the reported label."
                + "\nA. " + ACTIONS.get(allowed.get(0))
                + (allowed.size() > 1 ? "\nB. " + ACTIONS.get(allowed.get(1)) : "")
                + "\nAnswer with the option letter. On the next line, write one short sentence "
                + "describing the detector's finding. Use only the facts supplied above.";
        try {
            return parse(generate(prompt, 42, "Choice: "), allowed, fallback, finding);
        } catch (Exception | LinkageError failure) {
            Log.e(TAG, "Local advisory failed", failure);
            return fallback;
        }
    }

    String smoke() {
        return generate("A metal nut has a visible scratch. In one sentence, tell the inspector "
                + "to isolate the part for quality review.", 36, "").text;
    }

    private static Advice parse(Response response, List<String> allowed, Advice fallback, VisionEngine.Finding finding) {
        String output = response.text.split("<\\|", 2)[0].trim();
        Matcher match = CHOICE.matcher(output);
        if (!match.find()) return withTiming(fallback, response.firstTokenMs);
        int index = match.group(1).equalsIgnoreCase("A") ? 0 : 1;
        if (index >= allowed.size()) return withTiming(fallback, response.firstTokenMs);
        String action = ACTIONS.get(allowed.get(index));
        String[] lines = output.split("\\R", 2);
        String explanation = lines.length == 2 ? lines[1].trim() : "";
        if (explanation.length() >= 8 && explanation.length() <= 220
                && !explanation.contains("\n") && isGrounded(explanation, finding)) {
            return new Advice(action, explanation, "Llama / ExecuTorch", response.firstTokenMs);
        }
        return new Advice(action, fallback.explanation(),
                "Llama action / evidence summary", response.firstTokenMs);
    }

    private static boolean isGrounded(String explanation, VisionEngine.Finding finding) {
        String lower = explanation.toLowerCase(Locale.ROOT);
        if (UNSUPPORTED.matcher(lower).find()) return false;
        if ("Review".equals(finding.decision())) {
            return lower.contains("uncertain") || lower.contains("confidence") || lower.contains("reinspect")
                    || lower.contains("image") || lower.contains("verify");
        }
        return switch (finding.defect()) {
            case "Scratch" -> lower.contains("scratch");
            case "Bent" -> lower.contains("bent") || lower.contains("bend");
            case "Discoloration" -> lower.contains("color") || lower.contains("colour");
            case "Flipped" -> lower.contains("flip") || lower.contains("orientation");
            default -> false;
        };
    }

    private static Advice withTiming(Advice advice, double firstTokenMs) {
        return new Advice(advice.action, advice.explanation, advice.source, firstTokenMs);
    }

    private record Response(String text, double firstTokenMs) {}

    private Response generate(String input, int maxTokens, String assistantPrefix) {
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
                + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n" + assistantPrefix;
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
