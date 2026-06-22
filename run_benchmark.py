import requests
import time
import os

url = "http://127.0.0.1:8000/chat"

# Benchmark query suites
test_queries = [
    # CHAT - General Knowledge
    ("What is 15 * 18?", "CHAT"),
    ("Who wrote Romeo and Juliet?", "CHAT"),
    ("Explain recursion in simple terms.", "CHAT"),
    ("What is the capital of Japan?", "CHAT"),
    ("How does binary search work?", "CHAT"),

    # BUILD - Create Tools
    ("Create a tool named bmi_calculator that calculates BMI.", "BUILD"),
    ("Build a random password generator tool.", "BUILD"),
    ("Create a currency converter tool.", "BUILD"),
    ("Build a unit conversion tool for length measurements.", "BUILD"),
    ("Create a loan EMI calculator tool.", "BUILD"),

    # BUILD - Modify Tools
    ("Modify the weather tool to accept country codes.", "BUILD"),
    ("Update the bmi_calculator tool to return BMI category.", "BUILD"),
    ("Add temperature units support to the weather tool.", "BUILD"),
    ("Modify the password generator to support special characters.", "BUILD"),
    ("Update the EMI calculator to include interest breakdown.", "BUILD"),

    # BUILD - Delete Tools
    ("Delete the bmi_calculator tool.", "BUILD"),
    ("Remove the weather tool.", "BUILD"),
    ("Delete the password generator tool.", "BUILD"),

    # CHAT - Tool Execution
    ("Calculate BMI for 70kg and 175cm.", "CHAT"),
    ("Generate a random password of length 12.", "CHAT"),
    ("Convert 100 USD to INR.", "CHAT"),
    ("Convert 5 kilometers to meters.", "CHAT"),
    ("Calculate EMI for 500000 at 8% for 5 years.", "CHAT"),

    # CHAT - Existing Tool Usage
    ("What is the weather in Chennai?", "CHAT"),
    ("What is the weather in New York?", "CHAT"),
    ("What is the current time?", "CHAT"),

    # Ambiguous Queries
    ("I need something that can calculate body mass index.", "BUILD"),
    ("Can you make a tool that tells me the weather?", "BUILD"),
    ("I want a utility for generating passwords.", "BUILD"),

    # Adversarial / Edge Cases
    ("Create a tool.", "BUILD"),
    ("Modify a tool.", "BUILD"),
    ("Delete a tool.", "BUILD"),
    ("Tell me something interesting.", "CHAT"),
    ("Help me with programming.", "CHAT")
]

def check_server():
    try:
        requests.get("http://127.0.0.1:8000/")
        return True
    except requests.exceptions.ConnectionError:
        return False

def main():
    print("Checking if AdaptBot server is running...")
    if not check_server():
        print("Error: The AdaptBot server is NOT running. Please start it first using:")
        print("  uv run uvicorn app:app --reload")
        return

    print("Server is active. Starting benchmark suite...\n")
    
    for i, (query, expected_intent) in enumerate(test_queries, 1):
        print(f"[{i}/{len(test_queries)}] Sending Query: '{query}' ({expected_intent})")
        start = time.time()
        try:
            res = requests.post(url, json={"message": query})
            elapsed = time.time() - start
            if res.status_code == 200:
                data = res.json()
                print(f"  -> Response: {data.get('response')[:100]}...")
                print(f"  -> Path Taken: {data.get('path')}")
                print(f"  -> Latency: {elapsed:.2f}s\n")
            else:
                print(f"  -> Request Failed with status {res.status_code}\n")
        except Exception as e:
            print(f"  -> Error occurred: {e}\n")
        
        # Cooldown between requests
        time.sleep(1.5)

    print("Benchmark suite completed!")
    print("You can view the detailed results in: benchmarks.csv")
    
    if os.path.exists("benchmarks.csv"):
        print("\nLast 5 entries in benchmarks.csv:")
        with open("benchmarks.csv", "r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in lines[-5:]:
                print("  " + line.strip())

if __name__ == "__main__":
    main()
