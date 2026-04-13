import requests
import time
import subprocess
import os
import signal

def test_api():
    # Start the server
    proc = subprocess.Popen(
        ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8005"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )
    time.sleep(10) # Wait for startup

    try:
        # Test /agents
        print("Testing /agents...")
        r = requests.get("http://localhost:8005/agents")
        print(f"Status: {r.status_code}")
        if r.status_code == 200:
            print("Response successfully serialized!")
        else:
            print(f"Error: {r.text}")

        # Test /trades
        print("\nTesting /trades...")
        r = requests.get("http://localhost:8005/trades")
        print(f"Status: {r.status_code}")
        if r.status_code == 200:
            print("Response successfully serialized!")
        else:
            print(f"Error: {r.text}")

        # Test /scan-project
        print("\nTesting /scan-project...")
        r = requests.post("http://localhost:8005/scan-project", json={"symbol": "BTC_USDT"})
        print(f"Status: {r.status_code}")
        if r.status_code == 200:
            print("Response successfully serialized!")
            print(r.json())
        else:
            print(f"Error: {r.text}")

    finally:
        os.kill(proc.pid, signal.SIGTERM)

if __name__ == "__main__":
    test_api()
