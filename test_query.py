"""Helper script to test queries via command line."""

import requests
import sys
import json

def ask_question(question: str, api_url: str = "http://localhost:8000"):
    """Send a question to the QA endpoint."""
    
    print(f"\n❓ Question: {question}\n")
    
    response = requests.post(
        f"{api_url}/qa",
        json={"question": question}
    )
    
    if response.status_code == 200:
        result = response.json()
        
        print("=" * 60)
        print("🧠 SEARCH PLAN")
        print("=" * 60)
        print(result.get('plan', 'No plan generated'))
        
        print("\n" + "=" * 60)
        print("🔍 SUB-QUESTIONS")
        print("=" * 60)
        for i, sq in enumerate(result.get('sub_questions', []), 1):
            print(f"{i}. {sq}")
        
        print("\n" + "=" * 60)
        print("✅ ANSWER")
        print("=" * 60)
        print(result['answer'])
        print()
        
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_query.py 'Your question here'")
        sys.exit(1)
    
    ask_question(" ".join(sys.argv[1:]))