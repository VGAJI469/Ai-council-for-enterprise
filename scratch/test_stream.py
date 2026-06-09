import httpx
import sys

def test_stream():
    url = "http://localhost:8000/council/debate"
    params = {
        "motion": "Test pilot rollout of automated loan approval algorithms in Tier-2 branches.",
        "dti": 42.0,
        "creditScore": 720.0,
        "defaultProbability": 8.0
    }
    
    print("Connecting to debate streaming endpoint...")
    try:
        with httpx.stream("GET", url, params=params, timeout=120.0) as response:
            if response.status_code != 200:
                print(f"Failed with status: {response.status_code}")
                return
            
            print("Connected! Listening for SSE events...\n")
            # Limit printing to first few events to verify structure
            count = 0
            for line in response.iter_lines():
                if line:
                    print(line)
                    count += 1
                    if "consensus" in line or count > 50:
                        print("\nReached final consensus or event limit. Stopping test.")
                        break
    except Exception as e:
        print(f"Error testing stream: {e}")

if __name__ == "__main__":
    test_stream()
