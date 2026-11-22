package org.bharat.edgeinspect;

import android.content.ContentValues;
import android.content.Context;
import android.database.Cursor;
import android.database.sqlite.SQLiteDatabase;
import android.database.sqlite.SQLiteOpenHelper;
import android.graphics.RectF;

import java.io.IOException;
import java.io.Writer;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;

final class InspectionLog extends SQLiteOpenHelper {
    private static final String TABLE = "inspections";
    private static final String[] COLUMNS = {"id", "timestamp", "input_source", "decision", "defect",
            "confidence", "severity", "location", "area_fraction", "vision_ms", "action", "explanation",
            "advice_source", "first_token_ms"};

    InspectionLog(Context context) { super(context, "edge-inspections.db", null, 1); }

    @Override public void onCreate(SQLiteDatabase database) {
        database.execSQL("CREATE TABLE inspections (id INTEGER PRIMARY KEY AUTOINCREMENT, "
                + "timestamp TEXT NOT NULL, input_source TEXT NOT NULL, decision TEXT NOT NULL, "
                + "defect TEXT NOT NULL, confidence REAL NOT NULL, severity TEXT NOT NULL, location TEXT NOT NULL, "
                + "area_fraction REAL NOT NULL, vision_ms REAL NOT NULL, action TEXT NOT NULL, "
                + "explanation TEXT NOT NULL, advice_source TEXT NOT NULL, first_token_ms REAL NOT NULL)");
    }

    @Override public void onUpgrade(SQLiteDatabase database, int oldVersion, int newVersion) {
        throw new IllegalStateException("No inspection database migration defined");
    }

    long save(String source, VisionEngine.Finding finding, AdvisoryEngine.Advice advice) {
        ContentValues row = fields(advice);
        row.put("timestamp", Instant.now().toString());
        row.put("input_source", source);
        row.put("decision", finding.decision());
        row.put("defect", finding.defect());
        row.put("confidence", finding.confidence());
        row.put("severity", finding.severity());
        row.put("area_fraction", finding.area());
        row.put("vision_ms", finding.milliseconds());
        RectF box = finding.box();
        row.put("location", box == null ? "" : String.format(Locale.US, "%.4f,%.4f,%.4f,%.4f",
                box.left, box.top, box.right, box.bottom));
        return getWritableDatabase().insertOrThrow(TABLE, null, row);
    }

    void updateAdvice(long id, AdvisoryEngine.Advice advice) {
        getWritableDatabase().update(TABLE, fields(advice), "id = ?", new String[]{Long.toString(id)});
    }

    private static ContentValues fields(AdvisoryEngine.Advice advice) {
        ContentValues values = new ContentValues();
        values.put("action", advice.action());
        values.put("explanation", advice.explanation());
        values.put("advice_source", advice.source());
        values.put("first_token_ms", advice.firstTokenMs());
        return values;
    }

    record Entry(long id, String timestamp, String detail, String advice, String adviceSource) {}

    List<Entry> latest() {
        List<Entry> entries = new ArrayList<>();
        String[] columns = {"id", "timestamp", "decision", "defect", "input_source", "action", "advice_source"};
        try (Cursor rows = getReadableDatabase().query(TABLE, columns, null, null, null, null, "id DESC", "50")) {
            while (rows.moveToNext()) {
                entries.add(new Entry(rows.getLong(0), rows.getString(1),
                        rows.getString(2) + " · " + rows.getString(3) + " · " + rows.getString(4),
                        rows.getString(5), rows.getString(6)));
            }
        }
        return entries;
    }

    void export(Writer output) throws IOException {
        output.write(String.join(",", COLUMNS) + "\r\n");
        try (Cursor rows = getReadableDatabase().query(TABLE, COLUMNS, null, null, null, null, "id ASC")) {
            while (rows.moveToNext()) {
                for (int index = 0; index < COLUMNS.length; index++) {
                    if (index > 0) output.write(",");
                    output.write("\"" + rows.getString(index).replace("\"", "\"\"") + "\"");
                }
                output.write("\r\n");
            }
        }
    }
}
