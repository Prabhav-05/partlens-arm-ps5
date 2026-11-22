#!/bin/sh
# Run from the project root with one authorized Android device connected.
set -eu
package=org.bharat.edgeinspect
apk=${1:-app/build/outputs/apk/debug/app-debug.apk}
adb install -r "$apk"
adb shell run-as "$package" mkdir -p files
for model in tokenizer.json llama.pte; do
    input=".local/models/$model"
    test -s "$input"
    size=$(wc -c < "$input" | tr -d ' ')
    adb shell -T "run-as $package sh -c 'cat > files/$model.partial'" < "$input"
    adb shell "run-as $package sh -c 'test \"\$(stat -c %s files/$model.partial)\" -eq $size && mv files/$model.partial files/$model'"
done
adb shell am start -n "$package/.MainActivity"
