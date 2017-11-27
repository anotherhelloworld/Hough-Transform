import subprocess
import sys

for i in range(0, 181):
    subprocess.call(["./build/Debug/transform", "build/Debug/testRect" + str(i) + ".jpg"])