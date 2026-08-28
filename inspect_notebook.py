import json

path = "notebooks/real_estate_analysis_cleaned.ipynb"

with open(path, encoding="utf-8") as f:
    notebook = json.load(f)

for index in [89, 90, 91, 92, 93]:
    cell = notebook["cells"][index]

    print("\n" + "=" * 80)
    print(f"CELL {index}")
    print("=" * 80)

    for output in cell.get("outputs", []):

        output_type = output.get("output_type")

        if output_type == "stream":
            text = output.get("text", "")
            if isinstance(text, list):
                text = "".join(text)
            print(text)

        elif output_type in ["execute_result", "display_data"]:
            data = output.get("data", {})

            text = data.get("text/plain", "")

            if isinstance(text, list):
                text = "".join(text)

            if text:
                print(text)