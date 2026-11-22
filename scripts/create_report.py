"""Generate the Phase 1 technical report from training and device evidence."""

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.pagesizes import A4
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

INK = colors.HexColor("#132735")
TEXT = colors.HexColor("#294250")
TEAL = colors.HexColor("#007565")
SOFT = colors.HexColor("#EAF5F2")
LINE = colors.HexColor("#D9E4E7")
WHITE = colors.white
WIDTH = A4[0] - 40 * mm

STYLES = {
    "kicker": ParagraphStyle("kicker", fontName="Helvetica-Bold", fontSize=9,
                             leading=12, textColor=TEAL, spaceAfter=11),
    "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=31,
                            leading=35, textColor=INK, spaceAfter=11),
    "lead": ParagraphStyle("lead", fontName="Helvetica", fontSize=13, leading=19,
                           textColor=TEXT, spaceAfter=12),
    "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=15, leading=19,
                         textColor=INK, spaceBefore=13, spaceAfter=7, keepWithNext=True),
    "body": ParagraphStyle("body", fontName="Helvetica", fontSize=10.1, leading=15,
                           textColor=TEXT, spaceAfter=8),
    "small": ParagraphStyle("small", fontName="Helvetica", fontSize=8.5, leading=12,
                            textColor=TEXT, spaceAfter=5),
    "cell": ParagraphStyle("cell", fontName="Helvetica", fontSize=9.1, leading=12, textColor=TEXT),
    "cell_head": ParagraphStyle("cell_head", fontName="Helvetica-Bold", fontSize=9, leading=12, textColor=WHITE),
    "metric": ParagraphStyle("metric", fontName="Helvetica-Bold", fontSize=21, leading=26,
                             textColor=TEAL, alignment=TA_CENTER),
    "metric_label": ParagraphStyle("metric_label", fontName="Helvetica", fontSize=9, leading=12,
                                   textColor=TEXT, alignment=TA_CENTER),
}


def p(text, style="body"):
    return Paragraph(text, STYLES[style])


def grid(rows, widths):
    cells = [[p(escape(str(value)), "cell_head" if row == 0 else "cell")
              for value in values] for row, values in enumerate(rows)]
    table = Table(cells, colWidths=[WIDTH * ratio for ratio in widths], hAlign="LEFT", repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), INK),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, SOFT]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 9),
        ("RIGHTPADDING", (0, 0), (-1, -1), 9),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LINEBELOW", (0, -1), (-1, -1), 0.6, LINE),
    ]))
    return table


def footer(canvas, doc):
    width, _ = A4
    canvas.setStrokeColor(LINE)
    canvas.line(20 * mm, 17 * mm, width - 20 * mm, 17 * mm)
    canvas.setFillColor(TEXT)
    canvas.setFont("Helvetica", 8)
    canvas.drawString(20 * mm, 12 * mm, "EDGEINSPECT  |  ARM BHARAT AI-SOC CHALLENGE  |  PHASE 1")
    canvas.drawRightString(width - 20 * mm, 12 * mm, f"{doc.page:02d}")


def pct(value):
    return f"{100 * value:.1f}%"


def make_report(metrics, device, destination):
    counts = metrics["counts"]
    test = metrics["test"]
    labels = ["Good", "Scratch", "Bent", "Color", "Flip"]
    stat = Table([
        [p(pct(test["defect_precision"]), "metric"), p(pct(test["defect_recall"]), "metric"),
         p(pct(test["defect_mask_mean_iou"]), "metric")],
        [p("Defect precision", "metric_label"), p("Defect recall", "metric_label"),
         p("Defect mask mean IoU", "metric_label")],
    ], colWidths=[WIDTH / 3] * 3)
    stat.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), SOFT), ("TOPPADDING", (0, 0), (-1, 0), 10),
        ("BOTTOMPADDING", (0, -1), (-1, -1), 10),
    ]))
    story = [
        p("PROBLEM STATEMENT 5  /  TECHNICAL REPORT", "kicker"),
        p("EdgeInspect", "title"),
        p("Offline, on-device defect inspection and operator guidance for metal nuts.", "lead"),
        stat, Spacer(1, 7),
        p("Inspection workflow", "h2"),
        p("An operator centers one part in the rear camera or imports an image. The app displays a defect class, "
          "confidence, and a bounding region from the predicted mask. Tapping <b>Inspect part</b> saves one "
          "inspection automatically. Hold and Review records receive a local recommendation. History and CSV "
          "export use the same SQLite records."),
        p("Deployment architecture", "h2"),
        grid([
            ["Stage", "Implementation"],
            ["Capture and preprocess", "CameraX latest-frame analysis or local image import; centered square crop; 192 x 192 RGB tensor."],
            ["Vision and triage", "ExecuTorch 1.4.1 with XNNPACK on Arm CPU. The CNN emits five class logits and a 48 x 48 defect mask."],
            ["Advisory", "Llama 3.2 1B Instruct SpinQuant on a separate ExecuTorch worker. Inputs are class, severity, location, confidence, and decision."],
            ["Operator output", "Display Pass, Hold, or Review. The app validates a returned action ID and saves the explanation, action, and advisory source."],
        ], [0.25, 0.75]),
        p("Runtime boundary", "h2"),
        p("All inference, decisions, and storage occur inside the Android app. The APK declares camera permission. "
          "Model provisioning is a separate setup step. A default action is persisted first so the inspection "
          "remains usable if generation is interrupted."),
        PageBreak(),
        p("MODEL EVALUATION", "kicker"),
        p("Detection quality", "title"),
        p("Custom supervised evaluation on the MVTec AD Metal Nut category.", "lead"),
        grid([["Partition", *labels, "Total"]] + [
            [title, *[row.get(key, 0) for key in metrics["labels"]], sum(row.values())]
            for title, row in counts.items()
        ], [.19, .13, .15, .13, .13, .13, .14]),
        p("Protocol", "h2"),
        p("The supplied good training images were kept for training. The supplied good test images were divided "
          "evenly between validation and test. Each defect class was split about 60/20/20 with seed 42. "
          "Training uses class-balanced sampling, rotations, mirroring, and modest exposure changes. "
          "Checkpoint selection and triage cutoffs use the validation split. This protocol differs from the "
          "dataset's unsupervised benchmark."),
        p("Held-out confusion matrix", "h2"),
        grid([["Actual / predicted", *labels]] + [[label, *row] for label, row in zip(labels, test["confusion_matrix"])],
             [0.24, 0.15, 0.16, 0.15, 0.15, 0.15]),
        Spacer(1, 8),
        p(f"Raw five-class accuracy was <b>{pct(test['accuracy'])}</b> over {test['count']} images. "
          f"The triage policy produced {test['review_count']} Reviews and {test['false_passes_above_threshold']} false "
          "Pass on this host evaluation. Defect precision and recall collapse the four defect predictions into "
          "one positive class. Mask IoU averages only defective images."),
        p("Triage and severity", "h2"),
        p(f"The classification confidence cutoff is {metrics['class_threshold']:.2f}. Mask pixels use "
          f"{metrics['mask_threshold']:.2f}; a good prediction with a contiguous region covering "
          f"{pct(metrics['anomaly_review_threshold'])} of the crop or more goes to Review. Low, Medium and High "
          "severity correspond to region area below 1%, from 1% to below 5%, and at least 5%. Area is an "
          "image-space proxy with no dimensional calibration."),
        PageBreak(),
        p("DEVICE VALIDATION AND DELIVERY", "kicker"),
        p("On-device evidence", "title"),
        p(escape(device.get("summary", "The project targets Arm64 Android 14 and newer.")), "lead"),
        grid([["Check", "Observed result"]] +
             [[name, value] for name, value in device.get("checks", {}).items()], [.32, .68]),
        p("Known limits", "h2"),
        p("The held-out split has 29 images under controlled capture conditions. The classifier can miss scratches "
          "or discoloration and can flag good parts. Camera throughput depends on lighting, framing, handset load, "
          "and heat. A production study needs fixture-captured factory samples, dimensional severity criteria, "
          "a longer sustained run, and operator evaluation."),
        p("Reproduction", "h2"),
        p("The repository includes Android code, model training and export, checksum-verified asset downloads, and "
          "phone setup instructions. The release includes the APK and demonstration. Llama and the source dataset "
          "can be fetched separately."),
        p("References", "h2"),
        p('<link href="https://arm-education.github.io/Arm-Developer-Labs/Bharat_AI_SoC_2026_27.html" color="#007565">Arm PS5 problem statement</link>  |  '
          '<link href="https://www.mvtec.com/research-teaching/datasets/mvtec-ad" color="#007565">MVTec AD and CC BY-NC-SA 4.0 terms</link>', "small"),
        p('<link href="https://github.com/pytorch/executorch/tree/v1.4.1" color="#007565">ExecuTorch 1.4.1</link>  |  '
          '<link href="https://huggingface.co/software-mansion/react-native-executorch-llama-3.2" color="#007565">Llama 3.2 SpinQuant export</link>  |  '
          '<link href="https://github.com/Prabhav-05/edgeinspect-arm-ps5" color="#007565">Project source</link>', "small"),
    ]
    destination.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(destination), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=17 * mm, bottomMargin=22 * mm, title="EdgeInspect Phase 1 Technical Report",
                            subject="Arm Bharat AI-SoC Challenge, Problem Statement 5")
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=Path(".local/vision/metrics.json"))
    parser.add_argument("--device", type=Path, default=Path(".local/evidence/device_results.json"))
    parser.add_argument("--output", type=Path, default=Path("output/pdf/EdgeInspect-Technical-Report.pdf"))
    args = parser.parse_args()
    make_report(json.loads(args.metrics.read_text()), json.loads(args.device.read_text()), args.output)


if __name__ == "__main__":
    main()
