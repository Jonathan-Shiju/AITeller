import os
import json

def get_bank_account_info(query: str):
    """
    Retrieves bank account details by account number or name from dummy data.
    :param query: Account number or name (case-insensitive).
    :return: Account details or not found message.
    """
    data_path = os.path.join(
        os.path.dirname(__file__),
        "../utils/dummy_bank_data.json"
    )
    with open(data_path, "r") as f:
        accounts = json.load(f)
    for acc in accounts:
        if acc["account_number"] == query or acc["name"].lower() == query.lower():
            return acc
    return f"No account found for '{query}'."

