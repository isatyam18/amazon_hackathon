import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing import clean_business_name, clean_address, extract_numeric_tokens, extract_postal_code, preprocess_record

def run_tests():
    test_cases = [
        {
            "entity_id": "S1-001",
            "business_name": "Société Générale S.A.R.L.",
            "business_address": "12 Bd Haussmann, 75009 Paris",
            "country": "France",
        },
        {
            "entity_id": "S1-002",
            "business_name": "Infosys Technologies Pvt Ltd",
            "business_address": "Plot No 44, Electronic City, Hosur Rd, Bangalore, Karnataka 560100",
            "country": "India",
        },
        {
            "entity_id": "S1-003",
            "business_name": "Alphabet & Google Inc.",
            "business_address": "1600 Amphitheatre Pkwy, Ste 100, Mountain View, CA 94043",
            "country": "US",
        },
    ]

    for tc in test_cases:
        res = preprocess_record(tc)
        print("ID:", res["entity_id"], f"({res['country']})")
        print("  Clean name :", res["clean_name"])
        print("  Core name  :", res["core_name"])
        print("  Suffix     :", res["legal_suffix"])
        print("  Clean addr :", res["clean_address"])
        print("  Postal code:", res["postal_code"])
        print("  Numerics   :", extract_numeric_tokens(res["clean_address"]))
        print("-" * 50)

if __name__ == "__main__":
    run_tests()
