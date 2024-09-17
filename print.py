import signal
import sys

def custom_exit(signum, frame):
    with open('./abcd.txt', 'w') as file:
        file.write('abcd')
    print(f"Exiting script due to signal {signum}")
    sys.exit(0)

# 处理SIGTERM信号
signal.signal(signal.SIGTERM, custom_exit)
# 处理SIGINT信号
signal.signal(signal.SIGINT, custom_exit)

print("Script is running. Send SIGTERM or SIGINT to exit.")

try:
    while True:
        # 模拟脚本的主要逻辑
        pass
except KeyboardInterrupt:
    print("\nKeyboard interrupt detected. Exiting script...")
    custom_exit(None, None)
