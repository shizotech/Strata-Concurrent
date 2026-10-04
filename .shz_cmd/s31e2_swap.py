import io, sys

path = "src/program/generate.cpp"
with io.open(path, encoding="utf-8") as f:
    src = f.read()

start_marker = "        // ==================== S3.1e-2: THE CONCURRENT DRIVER ====================\n"
end_marker = "        // ---- S3.1e-2: everything below is 0.1.30's serial driver, unchanged."

i = src.find(start_marker)
j = src.find(end_marker)
if i < 0 or j < 0 or j <= i:
    print("MARKERS NOT FOUND", i, j)
    sys.exit(1)

with io.open(".shz_cmd/s31e2_driver.txt", encoding="utf-8") as f:
    new = f.read()

if not new.endswith("\n"):
    new += "\n"

out = src[:i] + new + src[j:]
with io.open(path, "w", encoding="utf-8") as f:
    f.write(out)
print("replaced %d bytes with %d bytes" % (j - i, len(new)))
