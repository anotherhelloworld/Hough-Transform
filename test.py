import subprocess
import sys

for i in range(0, 181):
    subprocess.call(["./DisplayImage", "testRect" + str(i) + ".jpg"])